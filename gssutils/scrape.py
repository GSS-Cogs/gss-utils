import json
import logging
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse

import html2text
import msgpack
import requests
from cachecontrol import CacheControl, serialize
from cachecontrol.caches.file_cache import FileCache
from cachecontrol.heuristics import LastModified
from lxml import html
from rdflib import BNode, URIRef

import gssutils.scrapers
from gssutils.metadata import PMDDataset, Excel, ODS, Catalog, ExcelOpenXML
from gssutils.utils import pathify


class BiggerSerializer(serialize.Serializer):

    def _loads_v4(self, request, data):
        try:
            cached = msgpack.loads(
                data, encoding='utf-8', max_bin_len=100*1000*1000) # 100MB
        except ValueError:
            return

        return self.prepare_response(request, cached)


class FilterError(Exception):
    """ Raised when filters don't uniquely identify a thing
    """

    def __init__(self, message):
        self.message = message


class MetadataError(Exception):
    """ Raised when a provided metadata info.json cannot be used
    """

    def __init__(self, message):
        self.message = message


def Scraper(uri_or_info, session=None):
    """
    Scraper wraps ScraperObj to allow us to depreciate the direct passing of uri's
    without breaking existing pipelines
    """
    if uri_or_info.endswith("info.json") or uri_or_info.endswith("info.schema.json"):

        try:
            with open(uri_or_info, "r") as f:
                info = json.load(f)
            uri = info["dataURL"]
        except Exception as e:
            raise MetadataError("Unable to acquire dataURL from the provided "
                                "info.json") from e

        return ScraperObj(uri, session, info=info)
    else:
        # It's an old style one, throw a depreciation warning then proceed
        logging.warning("The direct passing of uri's has been depreciated. Please "
                    "use the info.json file and pass in your dataURL there.")
        return ScraperObj(uri_or_info, session)


class ScraperObj:
    def __init__(self, uri, session, info=None):

        self.uri = uri
        self.dataset = PMDDataset()
        self.catalog = Catalog()
        self.dataset.modified = datetime.now()
        self.distributions = []
        self.info = info

        if session:
            self.session = session
        else:
            self.session = CacheControl(requests.Session(),
                                        cache=FileCache('.cache'),
                                        serializer=BiggerSerializer(),
                                        heuristic=LastModified())
        if 'JOB_NAME' in os.environ:
            self._base_uri = URIRef('http://gss-data.org.uk')
            if os.environ['JOB_NAME'].startswith('GSS/'):
                self._dataset_id = pathify(os.environ['JOB_NAME'][len('GSS/'):])
            else:
                self._dataset_id = pathify(os.environ['JOB_NAME'])
        else:
            self._base_uri = BNode()
            parsed_scrape_uri = urlparse(self.uri)
            self._dataset_id = parsed_scrape_uri.netloc.replace('.', '/') + parsed_scrape_uri.path
        self.update_dataset_uris()
        self._run()

    def _repr_markdown_(self):
        md = ""
        if hasattr(self.catalog, 'dataset') and len(self.catalog.dataset) > 1 and len(self.distributions) == 0:
            md = md + f'## {self.catalog.title}\n\nThis is a catalog of datasets; choose one from the following:\n\n'
            md = md + '\n'.join([f'* {d.label}' for d in self.catalog.dataset])
        else:
            if hasattr(self.dataset, 'label'):
                md = md + f'## {self.dataset.label}\n\n'
            if hasattr(self.dataset, 'comment'):
                md = md + f'{self.dataset.comment}\n\n'
            if hasattr(self.dataset, 'description'):
                md = md + f'### Description\n\n{self.dataset.description}\n\n'
            if len(self.distributions) > 0:
                md = md + "### Distributions\n\n"
                for d in self.distributions:
                    t = {Excel: 'MS Excel Spreadsheet', ODS: 'ODF Spreadsheet'}
                    if hasattr(d, 'issued'):
                        md = md + f"1. {d.title} ([{t.get(d.mediaType, d.mediaType)}]({d.downloadURL})) - {d.issued}\n"
                    else:
                        md = md + f"1. {d.title} ([{t.get(d.mediaType, d.mediaType)}]({d.downloadURL}))\n"

        return md

    @staticmethod
    def to_markdown(node):
        if type(node) == list:
            return html2text.html2text('\n'.join([html.tostring(n, encoding='unicode') for n in node]))
        else:
            return html2text.html2text(html.tostring(node, encoding='unicode'))

    def _run(self):
        page = self.session.get(self.uri)

        # TODO - the below should go into a bucket, we should be logging out a
        # nice clickable link to said bucket

        # Dump the scraped file locally if we're debugging
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            out_path = '{}/debug_scrape.html'.format(os.getcwd())
            logging.debug("Writing scrape as simple html to: " + out_path)
            with open(out_path, "w") as f:
                f.write(page.text)

        # TODO - not all scrapers will necessarily need the beautified HTML DOM
        tree = html.fromstring(page.text)
        scraped = False
        for start_uri, scrape in gssutils.scrapers.scraper_list:
            if self.uri.startswith(start_uri):

                # Scrape
                self.dataset.landingPage = self.uri
                scrape(self, tree)
                scraped = True

                # Before finishing, where we have an info.json, use it to plug any metadata gaps
                if self.info is not None:
                    self._populate_missing_metadata()
                break

        if not scraped:
            raise NotImplementedError(f'No scraper for {self.uri}')
        return self

    def _populate_missing_metadata(self):
        """
        Use the info.json file to populate any missing metadata fields.
        """

        try:
            # Dataset level metadata
            if not hasattr(self.dataset, 'title'): self.dataset.title = self.info["title"]
            if not hasattr(self.dataset, 'description'): self.dataset.description = self.info["description"]

        except Exception as e:
            raise MetadataError("Aborting. Issue encountered while attempting checking "
                                "the info.json for supplementary metadata.") from e


    @staticmethod
    def _filter_one(things, **kwargs):
        latest = kwargs.pop('latest', False)
        matches = [
            d for d in things if all(
                [v(d.__dict__[k]) if callable(v) else (hasattr(d, k) and d.__dict__[k] == v)
                 for k, v in kwargs.items()]
            )]
        if len(matches) > 1:
            if latest:
                return max(matches, key=lambda d: d.issued)
            else:
                raise FilterError('more than one match for given filter(s)')
        elif len(matches) == 0:
            raise FilterError('nothing matches given filter(s)')
        else:
            return matches[0]

    def select_dataset(self, **kwargs):
        dataset = Scraper._filter_one(self.catalog.dataset, **kwargs)
        self.dataset = dataset
        if not hasattr(self.dataset, 'description') and hasattr(self.catalog, 'description'):
            self.dataset.description = self.catalog.description
        self.dataset.modified = datetime.now() # TODO: decision on modified date
        self.update_dataset_uris()
        self.distributions = dataset.distribution

    def distribution(self, **kwargs):
        return Scraper._filter_one(self.distributions, **kwargs)

    def set_base_uri(self, uri):
        self._base_uri = uri
        self.update_dataset_uris()

    def set_dataset_id(self, id):
        self._dataset_id = id
        self.update_dataset_uris()

    def update_dataset_uris(self):
        self.dataset.uri = urljoin(self._base_uri, f'data/{self._dataset_id}')
        self.dataset.graph = urljoin(self._base_uri, f'graph/{self._dataset_id}/metadata')
        self.dataset.inGraph = urljoin(self._base_uri, f'graph/{self._dataset_id}')
        self.dataset.sparqlEndpoint = urljoin(self._base_uri, '/sparql')

    def set_family(self, family):
        self.dataset.family = family

    def set_theme(self, theme):
        self.dataset.theme = theme

    def set_description(self, description):
        self.dataset.description = description

    def generate_trig(self):
        return self.dataset.as_quads().serialize(format='trig')

    @property
    def title(self):
        return self.dataset.title

    @property
    def description(self):
        return self.dataset.description

    @property
    def publication_date(self):
        return self.dataset.issued.isoformat()

    @property
    def next_release(self):
        return self.dataset.updateDueOn.isoformat()

    @property
    def contact(self):
        return self.dataset.contactPoint
