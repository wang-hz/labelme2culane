import json
import logging
import shutil
from pathlib import Path

import cv2
import labelme


def main():
    src_dir = Path(r'../data/raw')
    src_image_suffix = '.png'
    seg_image_suffix = '.png'

    culane_dir = Path(r'../data/culane')
    list_dir = culane_dir / 'list'
    data_dir = culane_dir / 'data'
    seg_dir = culane_dir / 'segment'

    list_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    for src_label_path in sorted(src_dir.iterdir()):
        if not src_label_path.is_file() or src_label_path.suffix != '.json':
            continue
        src_image_path = src_dir / f'{src_label_path.stem}{src_image_suffix}'
        if not src_image_path.exists():
            logging.warning(f'{src_image_path} does not exist')
            continue

        with open(src_label_path, 'r') as src_label_file:
            src_label_data = json.load(src_label_file)
        lines = {shape['label']: shape['points'] for shape in src_label_data['shapes']}

        lines_count = len(lines)
        if lines_count > 4:
            logging.warning(f'the count of lanes is {lines_count} larger than 4')
            continue

        dst_image_path = data_dir / src_image_path.name
        shutil.copy(src_image_path, dst_image_path)

        dst_label_path = data_dir / f'{src_label_path.stem}.lines.txt'
        with open(dst_label_path, 'w') as dst_label_file:
            for key in sorted(lines.keys()):
                line = lines[key]
                flat_line = [str(x) for point in line for x in point]
                dst_label_file.write(' '.join(flat_line))
                dst_label_file.write('\n')

        image = cv2.imread(str(src_image_path))
        seg_image, _ = labelme.utils.labelme_shapes_to_label(image.shape, src_label_data['shapes'])
        dst_seg_path = seg_dir / f'{src_image_path.stem}{seg_image_suffix}'
        cv2.imwrite(str(dst_seg_path), seg_image)

        train_path = list_dir / 'train.txt'
        with open(train_path, 'w') as train_file:
            line = str(dst_image_path.relative_to(culane_dir))
            train_file.write(f'/{line}\n')

        flags = []
        for _ in range(lines_count):
            flags.append('1')
        for _ in range(4 - lines_count):
            flags.append('0')
        train_gt_path = list_dir / 'train_gt.txt'
        with open(train_gt_path, 'w') as train_gt_file:
            dir1 = str(dst_image_path.relative_to(culane_dir))
            dir2 = str(dst_seg_path.relative_to(culane_dir))
            train_gt_file.write(f'/{dir1} /{dir2} {" ".join(flags)}\n')


if __name__ == '__main__':
    main()
