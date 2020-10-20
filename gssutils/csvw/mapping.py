import csv
import gzip
import json
import logging
from io import TextIOBase
from pathlib import Path
from typing import List, Optional, Dict, TextIO, Any, Set, Union, NamedTuple
from urllib.parse import urljoin
import pandas as pd

from uritemplate import variables

from gssutils import pathify
from gssutils.csvw.dsd import DataSet, DimensionComponent, MeasureComponent, AttributeComponent, Component, \
    DimensionProperty, DSD, Resource, MeasureProperty, AttributeProperty
from gssutils.csvw.namespaces import prefix_map, URI
from gssutils.csvw.table import Column, TableSchema, Table, ForeignKey

default_map = {
    "Value": {
        "unit": "#unit/count",
        "measure": "#measure/total",
        "datatype": "integer"
    }
}

class MeasureDefinitionError(Exception):
    """ Raised when a multiple measures datacube has been indicated, but has not been correctly defined."""

    def __init__(self, message):
        self.message = message

class MapObject(NamedTuple):
    """
    A single instance of mapping for a given concept
    """
    name: str
    obj: dict

class CSVWMapping:
    def __init__(self, data_root="http://gss-data.org.uk/def"):
        self._csv_filename: Optional[URI] = None
        self._csv_stream: Optional[TextIO] = None
        self._mapping: Dict[str, Any] = {}
        self._column_names = List[str]
        self._columns: Dict[str, Column] = {}
        self._external_tables: List[Table] = []
        self._dataset_uri: Optional[URI] = None
        self._dataset = DataSet()
        self._components: List[Component] = []
        self._registry: Optional[URI] = None
        self._keys: List[str] = []
        self._metadata_filename: Optional[URI] = None
        self._dataset_uri: Optional[URI] = None
        self._foreign_keys: List[ForeignKey] = None
        self._has_multiple_measures: bool = False
        self._multiple_measures_map: List[MapObject] = []
        self._data_root: str = data_root

    @staticmethod
    def namify(column_header: str):
        return pathify(column_header).replace('-', '_')

    @staticmethod
    def classify(column_header: str):
        return ''.join(part.capitalize() for part in pathify(column_header).split('-'))

    def join_dataset_uri(self, relative: str):
        # treat the dataset URI as an entity that when joined with a fragment, just adds
        # the fragment, but when joined with a relative path, turns the dataset URI into a container
        # by adding a / to the end before adding the relative path
        if self._dataset_uri is None:
            return URI(relative)
        elif relative.startswith('#'):
            return URI(urljoin(self._dataset_uri, relative, allow_fragments=True))
        else:
            return URI(urljoin(self._dataset_uri + '/', relative, allow_fragments=True))

    def set_csv(self, csv_filename: URI):

        # csv and csv.gz need to be read in slightly different ways
        if str(csv_filename).endswith("csv"):
            with open(csv_filename, newline='', encoding='utf-8') as f:
                self.set_input(csv_filename, f)
        elif str(csv_filename).endswith("csv.gz"):
            with gzip.open(csv_filename, encoding='utf-8') as f:
                self.set_input(csv_filename, f)
        else:
            raise ValueError("Only csv types of .csv and /csv.gz are supported."
                    " Not {}".format(csv_filename))

    def set_input(self, filename: URI, stream: TextIO):
        self._csv_stream = stream
        self._csv_filename = Path(str(filename)[:-3]) if str(filename).endswith(".csv.gz") else filename
        reader = csv.DictReader(stream)
        self._column_names = reader.fieldnames
        for col in self._column_names:
            self._columns[col] = Column(name=CSVWMapping.namify(col), titles=col, datatype="string")

    def set_mapping(self, mapping):
        if 'transform' in mapping and 'columns' in mapping['transform']:
            self._mapping = mapping['transform']['columns']
        else:
            logging.error(f'No column mapping found.')

    def set_additional_foreign_key(self, foreign_key: ForeignKey):
        if self._foreign_keys is None:
            self._foreign_keys = []
        self._foreign_keys.append(foreign_key)

    def set_dataset_uri(self, uri: URI):
        self._dataset_uri = uri

    def set_registry(self, uri: URI):
        self._registry = uri

    def _validate(self):
        # check variable names are consistent
        declared_names = set([col.name for col in self._columns.values()])
        used_names: Set[str] = set()
        for name_set in (
            variables(t)
            for col in self._columns.values()
            for t in [col.propertyUrl, col.valueUrl]
            if t is not None
        ):
            used_names.update(name_set)
        if not declared_names.issuperset(used_names):
            logging.error(f"Unmatched variable names: {used_names.difference(declared_names)}")
        # check used prefixes
        used_prefixes = set(
            t.split(':')[0]
            for col in self._columns.values()
            for t in [col.propertyUrl, col.valueUrl]
            if t is not None and not t.startswith('http') and ':' in t
        )
        if not set(prefix_map.keys()).issuperset(used_prefixes):
            logging.error(f"Unknown prefixes used: {used_prefixes.difference(prefix_map.keys())}")

    def _prep_multicube(self):
        """
        If the csv represents a datacube with multiple measures, gather the additional attributes required
        to correctly generate the csvw.
        """

        # Confirm the csv has the columns that it should have
        required_columns = ["Measure Type", "Unit"]
        for req_col in required_columns:
            if req_col not in self._column_names:
                raise MeasureDefinitionError('To create a multiple measures dataset, you need to include " \
                    " all of these columns: "{}". You do not have "{}"'.format(",".join(required_columns), 
                    req_col))

        for col in ["Measure Type"]:
            unique_values = pd.read_csv(self._csv_filename , usecols=[col])[col].unique().tolist()
            self._multiple_measures_map += [MapObject(name="Value", obj={"measure":x}) for x in unique_values]

        self._has_multiple_measures = True

    def _create_csvw_measures(self, name=None, obj=None):
        """
        Creates the csvw components required to represent the measures and units used by the datacube.
        Triggered by two mutually exclusive scenarios:
        1.) A single measure dataset, indicated by passing both a name and an obj to this method.
        2.) A multiple measures datasets, indicated by passing neither,
        """

        print("REGISTRY IS: .... ", self._registry)

        if self._has_multiple_measures:
            map_objects = self._multiple_measures_map
        else:
            map_objects = [MapObject(name=name, obj=obj)] # Just the one measure to worry about

        for map_obj in map_objects:
            if self._has_multiple_measures:
                self._columns[map_obj.name] = self._columns[map_obj.name]._replace(propertyUrl=URI("{}/measure/{{measure_type}}".format(self._data_root)))
            else:
                self._columns[map_obj.name] = self._columns[map_obj.name]._replace(propertyUrl=map_obj.obj["measure"])

            if "datatype" in map_obj.obj:
                self._columns[map_obj.name] = self._columns[map_obj.name]._replace(datatype=map_obj.obj["datatype"])
            else:
                self._columns[map_obj.name] = self._columns[map_obj.name]._replace(datatype="number")

            self._components.extend([
                DimensionComponent(
                    at_id=self.join_dataset_uri("#component/measure_type"),
                    qb_componentProperty=Resource(at_id=URI("http://purl.org/linked-data/cube#measureType")),
                    qb_dimension=DimensionProperty(
                        at_id=URI("http://purl.org/linked-data/cube#measureType"),
                        rdfs_range=Resource(at_id=URI("http://purl.org/linked-data/cube#MeasureProperty"))
                    )
                ),
                MeasureComponent(
                    at_id=self.join_dataset_uri(f"#component/{pathify(map_obj.name)}"),
                    qb_componentProperty=Resource(at_id=map_obj.obj["measure"]),
                    qb_measure=MeasureProperty(at_id=map_obj.obj["measure"])
                ),
                AttributeComponent(
                    at_id=self.join_dataset_uri(f"#component/unit"),
                    qb_componentProperty=Resource(
                        at_id=URI("http://purl.org/linked-data/sdmx/2009/attribute#unitMeasure")
                    ),
                    qb_attribute=AttributeProperty(
                        at_id=URI("http://purl.org/linked-data/sdmx/2009/attribute#unitMeasure")
                    )
                )
            ])

        # Create virtual measures and units
        self._columns["virt_unit"] = Column(
            name="virt_unit",
            virtual=True,
            propertyUrl=URI("http://purl.org/linked-data/sdmx/2009/attribute#unitMeasure"),
            valueUrl=URI(obj["unit"]) if not self._has_multiple_measures else URI("{}/concept/measurement-units/{{unit}}".format(self._data_root))
        )
        self._columns["virt_measure"] = Column(
            name="virt_measure",
            virtual=True,
            propertyUrl=URI("http://purl.org/linked-data/cube#measureType"),
            valueUrl=URI(obj["measure"]) if not self._has_multiple_measures else URI("{}/measure/{{measure_type}}".format(self._data_root))
        )

    def _as_csvw_object(self):

        # Is this a multi measure dataset?
        if "Measure Type" in self._column_names or "Unit" in self._column_names:
            self._prep_multicube()

        # If multiple measures, we don't map the value column from the csv
        if self._has_multiple_measures:
            self._column_names = [x for x in self._column_names if x != "Value"]

        objects_to_map = []
        for name in self._column_names:
            # If the column name appears in the map, we've some sort of definition of what it is
            # so create a 'MapObject' for processing later.
            if self._mapping is not None and name in self._mapping and isinstance(self._mapping[name], dict):
                map_obj = MapObject(name=name, obj=self._mapping[name])
                objects_to_map.append(map_obj)
            else:
                # otherwise assume local dimension, with optional definition
                description: Optional[str] = None
                if self._mapping is not None and name in self._mapping and isinstance(self._mapping[name], str):
                    description = self._mapping[name]
                self._keys.append(self._columns[name].name)
                self._columns[name] = self._columns[name]._replace(
                    propertyUrl=self.join_dataset_uri(f"#dimension/{pathify(name)}"),
                    valueUrl=self.join_dataset_uri(f"#concept/{pathify(name)}/{{{self._columns[name].name}}}")
                )
                self._components.append(DimensionComponent(
                    at_id=self.join_dataset_uri(f"#component/{pathify(name)}"),
                    qb_componentProperty=Resource(at_id=self.join_dataset_uri(f"#dimension/{pathify(name)}")),
                    qb_dimension=DimensionProperty(
                        at_id=self.join_dataset_uri(f"#dimension/{pathify(name)}"),
                        rdfs_range=Resource(
                            at_id=self.join_dataset_uri(f"#class/{CSVWMapping.classify(name)}")
                        ),
                        qb_codeList=Resource(
                            at_id = self.join_dataset_uri(f"#scheme/{pathify(name)}")
                        ),
                        rdfs_label=name,
                        rdfs_comment=description
                    )
                ))
        
        for map_obj in objects_to_map:

            # TODO - call these directly, i.e "map_obj.name" etc
            name = map_obj.name
            obj = map_obj.obj

            if "dimension" in obj and "value" in obj:
                self._keys.append(self._columns[name].name)
                self._columns[name] = self._columns[name]._replace(
                    propertyUrl=URI(obj["dimension"]),
                    valueUrl=URI(obj["value"])
                )
                self._components.append(DimensionComponent(
                    at_id=self.join_dataset_uri(f"#component/{pathify(name)}"),
                    qb_componentProperty=Resource(at_id=URI(obj["dimension"])),
                    qb_dimension=DimensionProperty(
                        at_id=URI(obj["dimension"]),
                        rdfs_range=Resource(
                            at_id=self.join_dataset_uri(f"#class/{CSVWMapping.classify(name)}")
                        )
                    )
                ))
            elif "parent" in obj and "value" in obj:
                # a local dimension that has a super property
                description: Optional[str] = None
                if "description" in obj:
                    description = obj["description"]
                source: Optional[URI] = None
                if "source" in obj:
                    source = URI(obj["source"])
                self._keys.append(self._columns[name].name)
                self._columns[name] = self._columns[name]._replace(
                    propertyUrl=self.join_dataset_uri(f"#dimension/{pathify(name)}"),
                    valueUrl=URI(obj["value"])
                )
                self._components.append(DimensionComponent(
                    at_id=self.join_dataset_uri(f"#component/{pathify(name)}"),
                    qb_componentProperty=Resource(at_id=self.join_dataset_uri(f"#dimension/{pathify(name)}")),
                    qb_dimension=DimensionProperty(
                        at_id=self.join_dataset_uri(f"#dimension/{pathify(name)}"),
                        rdfs_range=Resource(
                            at_id=self.join_dataset_uri(f"#class/{CSVWMapping.classify(name)}")
                        ),
                        qb_codeList=Resource(
                            at_id=self.join_dataset_uri(f"#scheme/{pathify(name)}")
                        ),
                        rdfs_label=name,
                        rdfs_comment=description,
                        rdfs_subPropertyOf=Resource(at_id=URI(obj["parent"])),
                        rdfs_isDefinedBy=Resource(at_id=source)
                    )
                ))
            elif "description" in obj:
                # local dimension with a definition and maybe source of the definition
                source: Optional[URI] = None
                if "source" in obj:
                    source = URI(obj["source"])
                self._keys.append(self._columns[name].name)
                self._columns[name] = self._columns[name]._replace(
                    propertyUrl=self.join_dataset_uri(f"#dimension/{pathify(name)}"),
                    valueUrl=self.join_dataset_uri(f"#concept/{pathify(name)}/{{{self._columns[name].name}}}")
                )
                self._components.append(DimensionComponent(
                    at_id=self.join_dataset_uri(f"#component/{pathify(name)}"),
                    qb_componentProperty=Resource(at_id=self.join_dataset_uri(f"#dimension/{pathify(name)}")),
                    qb_dimension=DimensionProperty(
                        at_id=self.join_dataset_uri(f"#dimension/{pathify(name)}"),
                        rdfs_range=Resource(
                            at_id=self.join_dataset_uri(f"#class/{CSVWMapping.classify(name)}")
                        ),
                        qb_codeList=Resource(
                            at_id=self.join_dataset_uri(f"#scheme/{pathify(name)}")
                        ),
                        rdfs_label=name,
                        rdfs_comment=obj["description"],
                        rdfs_isDefinedBy=Resource(at_id=source)
                    )
                ))

            elif "attribute" in obj and "value" in obj:
                self._columns[name] = self._columns[name]._replace(
                    propertyUrl=URI(obj["attribute"]),
                    valueUrl=URI(obj["value"])
                )
                self._components.append(AttributeComponent(
                    at_id=self.join_dataset_uri(f"#component/{pathify(name)}"),
                    qb_componentProperty=Resource(at_id=URI(obj["attribute"])),
                    qb_attribute=AttributeProperty(
                        at_id=URI(obj["attribute"]),
                        rdfs_range=Resource(
                            at_id=self.join_dataset_uri(f"#class/{CSVWMapping.classify(name)}")
                        )
                    )
                ))
            elif "unit" in obj and "measure" in obj:
                if self._has_multiple_measures:
                    raise MeasureDefinitionError('Aborting. You can either define a combination of ' \
                        '"measure" and "unit" via a "Value" field in the mapping, or include a ' \
                        'Measure Type" and "Unit" column in the csv. You cannot do both.')
                self._create_csvw_measures(name=name, obj=obj)

        # If the cube has multiple measures, go make them
        if self._has_multiple_measures:
            self._create_csvw_measures()
            
        self._columns["virt_dataset"] = Column(
            name="virt_dataset",
            virtual=True,
            propertyUrl=URI("qb:dataSet"),
            valueUrl=URI(self.join_dataset_uri("#dataset"))
        )
        self._columns["virt_type"] = Column(
            name="virt_type",
            virtual=True,
            propertyUrl=URI("rdf:type"),
            valueUrl=URI("qb:Observation")
        )

        self._validate()
        return {
            "@context": ["http://www.w3.org/ns/csvw", {"@language": "en"}],
            "tables": self._as_tables(),
            "@id": self.join_dataset_uri("#tables"),
            "prov:hadDerivation": DataSet(
                at_id=self.join_dataset_uri('#dataset'),
                qb_structure=DSD(
                    at_id=self.join_dataset_uri('#structure'),
                    qb_component=self._components
                )
            )
        }

    def _as_tables(self):
        table_uri = URI(Path(self._csv_filename).name)  # default is that metadata is filename + '-metadata.json'
        if self._metadata_filename is not None:
            table_uri = URI(self._csv_filename.relative_to(self._metadata_filename.parent))
            
        # TODO - dry
        if self._foreign_keys is not None:
            main_table = Table(
                url=table_uri,
                tableSchema=TableSchema(
                    columns=list(self._columns.values()),
                    primaryKey=self._keys,
                    aboutUrl=self.join_dataset_uri('/'.join('{' + s + '}' for s in self._keys)),
                    foreignKeys=self._foreign_keys
                )
            )
        else:
            main_table = Table(
                url=table_uri,
                tableSchema=TableSchema(
                    columns=list(self._columns.values()),
                    primaryKey=self._keys,
                    aboutUrl=self.join_dataset_uri('/'.join('{' + s + '}' for s in self._keys))
                )
            )
        return self._external_tables + [main_table]

    @staticmethod
    def _as_plain_obj(o):
        def fix_prefix(key: str):
            for prefix, replace in {'at_': '@', 'qb_': 'qb:', 'rdfs_': 'rdfs:'}.items():
                if key.startswith(prefix):
                    return replace + key[len(prefix):]
            return key
        if isinstance(o, tuple):
            try:
                return {fix_prefix(k): CSVWMapping._as_plain_obj(v) for (k, v) in dict(o._asdict()).items() if v is not None}
            except AttributeError:
                return o
        elif isinstance(o, dict):
            return {k: CSVWMapping._as_plain_obj(v) for (k, v) in o.items()}
        elif isinstance(o, Path):
            return str(o)
        elif isinstance(o, list):
            return [CSVWMapping._as_plain_obj(i) for i in o]
        else:
            return o

    def write(self, out: Union[URI, TextIO]):
        if not isinstance(out, TextIOBase):
            self._metadata_filename = out
            stream = open(out, "w", encoding="utf-8")
        else:
            stream = out
        plain_obj = CSVWMapping._as_plain_obj(self._as_csvw_object())
        logging.debug(json.dumps(plain_obj, indent=2))
        json.dump(plain_obj, stream, indent=2)
