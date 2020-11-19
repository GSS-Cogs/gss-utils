import re
from urllib.parse import urljoin

from dateutil.parser import parse
from lxml import html

from gssutils.metadata import GOV
from gssutils.metadata.dcat import Distribution
from gssutils.metadata.pmdcat import Dataset


def scrape(scraper, tree):
    page_type = tree.xpath(
        "//span[contains(concat(' ', @class, ' '), ' article-header__label ')]/text()")[0]

    if page_type.strip() == 'Series / Collection':
        scraper.catalog.title = tree.xpath("//h1/text()")[0]
        scraper.catalog.uri = scraper.uri + '#catalog'
        scraper.catalog.publisher = GOV['nhs-digital']
        scraper.catalog.license = 'http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/'
        scraper.catalog.rights = 'https://digital.nhs.uk/about-nhs-digital/terms-and-conditions'
        scraper.catalog.comment = ' '.join(tree.xpath(
            "//div[@id='section-summary']/div[@itemprop='description']/*/text()"))
        scraper.catalog.dataset = []

        articles = tree.xpath("//article[@class='cta']")
        for article in articles:
            dataset = Dataset(scraper.uri)
            dataset.distribution = []
            dataset.publisher = scraper.catalog.publisher
            dataset.license = scraper.catalog.license
            article_link = article.xpath('descendant::a')[0]
            dataset.title = article_link.get('title')
            href = article_link.get('href')
            dataset.landingPage = urljoin(scraper.uri, href)
            article_tree = html.fromstring(
                scraper.session.get(
                    dataset.landingPage).text)
            article_type = article_tree.xpath(
                "//span[contains(concat(' ', @class, ' '), ' article-header__label ')]/text()")[0]

            assert article_type.startswith(
                'Publication'), 'Expecting publication'

            details_node = article_tree.xpath("//dl[@class='detail-list']")[0]
            details = {}

            for node in details_node:
                if node.tag == 'dt' and node.get(
                        'class') == 'detail-list__key':
                    key = node.text.strip().lower()
                    if key.endswith(':'):
                        key = key[:-1].strip()
                elif node.tag == 'dd' and node.get('class') == 'detail-list__value':
                    value = node.text.strip()
                    if key not in details:
                        details[key] = [value]
                    else:
                        details[key].append(value)

            if 'publication date' in details:
                dataset.issued = parse(
                    details['publication date'][0], dayfirst=True)

            # Todo: spatiotemporal coverage and resolution/granularity
            # Todo: national statistics / official statistics badges

            resources = article_tree.xpath(
                "//ul[@data-uipath='ps.publication.resources-attachments']/li/a")

            for link in resources:
                dist = Distribution(scraper)
                dist.title = link.get('title')

                if hasattr(dataset, 'issued'):
                    dist.issued = dataset.issued

                dist.downloadURL = urljoin(
                    dataset.landingPage, link.get('href'))
                file_data = link.xpath("div[@class='block-link__body']")[0]
                dist.mediaType = str(file_data.xpath("meta/@content")[0])
                size = file_data.xpath(
                    "span/span[@class='fileSize']/span[@itemprop='contentSize']/text()")[0]
                size_match = re.match(
                    r'([0-9]+(\.[0-9]*)?)\s*(kB|MB|GB)', size)

                if size_match and size_match.group(3) == 'kB':
                    # https://en.wikipedia.org/wiki/Kilobyte kB = 1000 while KB = 1024
                    dist.byteSize = int(float(size_match.group(1)) * 1000)
                elif size_match and size_match.group(3) == 'MB':
                    # https://en.wikipedia.org/wiki/Megabyte MB = 10^6 bytes
                    dist.byteSize = int(float(size_match.group(1)) * 1000000)
                elif size_match and size_match.group(3) == 'GB':
                    # https://en.wikipedia.org/wiki/Gigabyte GB = 10^9 bytes, GiB = 2^30 bytes
                    dist.byteSize = int(
                        float(size_match.group(1)) * 1000000000)
                dataset.distribution.append(dist)

            scraper.catalog.dataset.append(dataset)
