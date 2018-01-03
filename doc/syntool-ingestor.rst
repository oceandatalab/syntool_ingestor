syntool-ingestor
================
The syntool-ingestor command generates a web-compatible representation from the
output of syntool-converter (GeoTIFF or NetCDF file). In addition to the data
representation (which depends on the chosen output format), syntool-ingestor
creates a metadata.json file which contains all the information needed to
determine if the granule matches a set of time and space criteria.

::
    syntool-ingestor -c CONFIG [--input-format INPUT_FMT]
                     [--input-options INPUT_OPTS] [--output-format OUTPUT_FMT]
                     [--output-options OUTPUT-OPTS] [--options-file OPT_FILE]
                     [--output-dir OUTPUT_DIR] INPUT


+-------------------------------+-----------------------+--------------------------------------------------------+
| Parameter                     | Format                | Description                                            |
+-------------------------------+-----------------------+--------------------------------------------------------+
| INPUT                         | path                  | NetCDF or GeoTIFF file from which data representations |
|                               |                       | must be generated.                                     |
+-------------------------------+-----------------------+--------------------------------------------------------+
| -c CONFIG                     | path                  | The data representations will be generated to fit the  |
|                               |                       | Syntool portal described in the CONFIG configuration   |
|                               |                       | file (projection, extent and viewport).                |
+-------------------------------+-----------------------+--------------------------------------------------------+
| --output-dir OUTPUT_DIR       | path                  | Save results in the OUTPUT directory (will be created  |
|                               |                       | if needed).                                            |
+-------------------------------+-----------------------+--------------------------------------------------------+
| --input-format INPUT_FMT      | string                | Format of the file provided as INPUT. Either “geotiff” |
|                               |                       | (GeoTIFF) or “idf” (NetCDF).                           |
+-------------------------------+-----------------------+--------------------------------------------------------+
| --input-options INPUT_OPTS    | key=value [key=value] | INPUT_FMT-specific options that will be passed to the  |
|                               |                       | data reader method.                                    |
|                               |                       | INPUT_OPTS is a list of key-value couples (format is   |
|                               |                       | “key=value”) separated by a blank space.               |
+-------------------------------+-----------------------+--------------------------------------------------------+
| --output-format OUTPUT_FMT    | string                | Format of the data representations. Either             |
|                               |                       | “rastertiles”, “vectorfield”, “raster”,                |
|                               |                       | “geojson_trajectory” or “trajectorytiles”.             |
+-------------------------------+-----------------------+--------------------------------------------------------+
| --output-options OUTPUT_OPTS  | key=value [key=value] | Options passed to the plugin in charge of formatting   |
|                               |                       | the output. Options must match the key=value pattern.  |
+-------------------------------+-----------------------+--------------------------------------------------------+
| --options-file OPT_FILE       | path                  | Read --input-format, --input-options, --output-format  |
|                               |                       | and --output-options from the OPT_FILE text file.      |
+-------------------------------+-----------------------+--------------------------------------------------------+
| -h, --help                    |                       | Display help message.                                  |
+-------------------------------+-----------------------+--------------------------------------------------------+


The configuration file for a portal in Web Mercator projection with a global viewport should be

::
    [portal]
    projection = 3857
    viewport = -20037508.34 -20037508.34 20037508.34 20037508.34
    extent = -20037508.34 -20037508.34 20037508.34 20037508.34

