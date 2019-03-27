import argparse
import csv
import json
from codecs import iterdecode
from pathlib import Path
from urllib import request, parse
from gssutils.utils import pathify


class CSVWSchema:

    def __init__(self, ref_base):
        self._ref_base = ref_base
        self._col_def = CSVWSchema._csv_lookup(
            parse.urljoin(ref_base, 'columns.csv'), 'title')
        self._comp_def = CSVWSchema._csv_lookup(
            parse.urljoin(ref_base, 'components.csv'), 'Label')
        self._codelists = {}
        for table in json.load(request.urlopen(parse.urljoin(ref_base, 'codelists-metadata.json')))['tables']:
            codelist_url = f'http://gss-data.org.uk/def/concept-scheme/{pathify(table["rdfs:label"])}'
            self._codelists[codelist_url] = table
        # need to resolve ref_common against relative URIs

    @staticmethod
    def _csv_lookup(url, key):
        stream = request.urlopen(url)
        reader = csv.DictReader(iterdecode(stream, 'utf-8'))
        return {row[key]: row for row in reader}

    def create(self, csv_filename, schema_filename):
        with open(csv_filename) as csv_io:
            with open(schema_filename, 'w') as schema_io:
                self.create_io(csv_io, schema_io, str(csv_filename.relative_to(schema_filename.parent)))

    def create_io(self, csv_io, schema_io, csv_url):
        schema_columns = []
        schema_tables = []
        schema_references = []
        schema_keys = []
        reader = csv.reader(csv_io)
        columns = next(reader)
        for column in columns:
            if column in self._col_def:
                column_def = self._col_def[column]
                is_unit = column_def['property_template'] == 'http://purl.org/linked-data/sdmx/2009/attribute#unitMeasure'
                column_schema = {
                    'titles': column,
                    'required': is_unit or (column_def['component_attachment'] not in ['qb:attribute']),
                    'name': column_def['name']
                }
                if 'regex' in column_def and column_def['regex'] not in (None, ''):
                    if column_def['datatype'] != 'string':
                        print(f"Column definition has regular expression guard '{column_def['regex']}' but datatype is '{column_def['datatype']}'")
                    else:
                        column_schema['datatype'] = {
                            'format': column_def['regex']
                        }
                else:
                    column_schema['datatype'] = column_def['datatype']
                schema_columns.append(column_schema)
                if column in self._comp_def:
                    component_def = self._comp_def[column]
                    codelist = component_def['Codelist']
                    if codelist in self._codelists:
                        reference = parse.urljoin(self._ref_base,
                                                  self._codelists[component_def['Codelist']]['url'])
                        schema_tables.append({
                            'url': reference,
                            'tableSchema': self._codelists[component_def['Codelist']]['tableSchema']
                        })
                        schema_references.append({
                            'columnReference': column_def['name'],
                            'reference': {
                                'resource': reference,
                                'columnReference': 'notation'
                            }
                        })
                    elif codelist.startswith('http://gss-data.org.uk/def/concept-scheme'):
                        print(f"Potentially missing concept scheme <{codelist}>")
                if is_unit or (column_def['component_attachment'] not in ['', 'qb:attribute']):
                    schema_keys.append(column_def['name'])
            else:
                print(f'"{column}" not defined')

        schema_tables.append({
            "url": csv_url,
            "tableSchema": {
                "columns": schema_columns,
                "foreignKeys": schema_references,
                "primaryKey": schema_keys
            }
        })

        schema = {
            "@context": ["http://www.w3.org/ns/csvw", {"@language": "en"}],
            "tables": schema_tables
        }

        json.dump(schema, schema_io, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Create CSV schema')
    parser.add_argument(
        'config_base_url',
        help='Base URL for table2qb configuration files, columns.csv, components,csv and codelists-metadata.json'
    )
    parser.add_argument('csv_file', type=argparse.FileType('r'),
                        help='Input CSV file with headers matching definitions in columns.csv.')
    parser.add_argument('schema_file', type=argparse.FileType('w'),
                        help='Output JSON file for use by CSVW validation tool, e.g. csvlint.')
    args = parser.parse_args()
    schema = CSVWSchema(args.config_base_url)
    schema.create_io(
        args.csv_file,
        args.schema_file,
        str(Path(args.csv_file.name).relative_to(Path(args.schema_file.name).parent)))
