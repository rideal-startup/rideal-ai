import shutil
from pathlib import Path

import click

import rideal.data.transform_raw_data

SENSORS_TO_EXCLUDE = ('com.samsung.sensor.grip',)


def _process_file(path: Path) -> str:
    first_line = True
    new_lines = []
    f = path.open()
    
    for line in f:
        time, sensor, data = line.split(',', 2)

        if first_line:
            first_line = False
            if sensor == "activityrecognition":
                time = "0"
        
        if not time.lstrip('-').isnumeric():
            print(f'Skipping {str(path)} due to incorrect time')
            f.close()
            return
        
        time = abs(int(time))
        if sensor not in SENSORS_TO_EXCLUDE:
            new_lines.append(','.join([str(time), sensor, data]))
    
    f.close()
    return ''.join(new_lines)

@click.command()
@click.option('--raw-path',
              help='Raw data directory',
              type=click.Path(file_okay=False, exists=True))
@click.option('--clean-path',
              help='Directory to store the cleaned data',
              type=click.Path(file_okay=False))
def main(raw_path, clean_path):
    raw_path = Path(raw_path)
    clean_path = Path(clean_path)

    if clean_path.exists():
        print('Deleting existing cleaned data')
        shutil.rmtree(str(clean_path))

    clean_path.mkdir(exist_ok=True, parents=True)

    for user_path in raw_path.iterdir():
        user = user_path.stem
        for f_name in user_path.glob('*.csv'):
            res = _process_file(f_name)
            if res:
                dst_path = clean_path/(f_name.stem + '.csv')
                dst_path.open('w').write(res)


if __name__ == "__main__":
    main()