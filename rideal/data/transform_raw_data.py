import shutil
from pathlib import Path

import click


# compute magnitude
DESIRED_SENSORS = ['android.sensor.accelerometer', 'android.sensor.orientation',
                   'android.sensor.linear_acceleration', 'android.sensor.gyroscope',
                   'android.sensor.magnetic_field', 'android.sensor.magnetic_field_uncalibrated',
                   'android.sensor.gyroscope_uncalibrated', 'android.sensor.gravity']


SENSOR_TO_TAKE_FIRST = ['android.sensor.light', 'speed', 'android.sensor.proximity', 'android.sensor.pressure']


def _process_file(path: Path) -> str:
    f = path.open()
    new_lines = []

    for line in f:
        time, sensor, data = line.strip().split(',', 2)
        
        if sensor in DESIRED_SENSORS:
            if sensor not in SENSOR_TO_TAKE_FIRST:
                new_lines.append(f'{time},{sensor},{data}')
            else:
                value = data.split(',')[0]
                new_lines.append(f'{time},{sensor},{value}')
    f.close()
    return '\n'.join(new_lines)


@click.command()
@click.option('--raw-path',
              help='Raw data directory',
              type=click.Path(file_okay=False, exists=True))
@click.option('--transform-path',
              help='Directory to store the cleaned data',
              type=click.Path(file_okay=False))
def main(raw_path, transform_path):
    raw_path = Path(raw_path)
    transform_path = Path(transform_path)

    if transform_path.exists():
        print('Deleting existing cleaned data')
        shutil.rmtree(str(transform_path))

    transform_path.mkdir(exist_ok=True, parents=True)

    for f_path in raw_path.glob('*.csv'):
        res = _process_file(f_path)
        if res:
            dst_path = transform_path/(f_path.stem + '.csv')
            dst_path.open('w').write(res)

if __name__ == "__main__":
    main()