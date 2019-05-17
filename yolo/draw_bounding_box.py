import numpy as np
from PIL.ImageDraw import ImageDraw
from PIL import ImageFont
from data_synthesis import array_to_image
from dataset_utils import index_to_class


def visualize_detection(a, boxes, labels):
    image = array_to_image(a)
    canvas = ImageDraw(image)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)

    for i in range(len(boxes)):
        xc, yc, w, h = boxes[i]
        label = labels[i]
        x = int(round(xc - w / 2))
        y = int(round(yc - h / 2))

        xy = [(x, y), (x + w, y + h)]
        canvas.rectangle(xy, width=2, outline=128)
        canvas.text((x + 2, y + 2), font=fnt, text=label, fill=255)

    image.show()


def draw_boxes(a, grid_size, output_volume, p_treshold=0.1):
    height, width = a.shape
    cell_height = height / grid_size
    cell_width = width / grid_size

    boxes = []
    labels = []

    for row in range(grid_size):
        for col in range(grid_size):
            detection_score = output_volume[row, col, 0]
            if detection_score > p_treshold:
                cell_x = col * cell_width
                cell_y = row * cell_height
                box = output_volume[row, col][1:5]

                predictions = output_volume[row, col][5:]
                index = np.argmax(detection_score * predictions)
                predicted_text = index_to_class[index]

                xc, yc, w, h = box

                xc_abs = cell_x + xc * cell_width
                yc_abs = cell_y + yc * cell_height

                w_abs = int(round(w * cell_width))
                h_abs = int(round(h * cell_height))

                boxes.append((xc_abs, yc_abs, w_abs, h_abs))
                labels.append(predicted_text)

    visualize_detection(a, boxes, labels)


if __name__ == '__main__':
    import os
    from yolo.yolo_home import YoloDatasetHome

    yolo_dir = '../datasets/yolo_dataset'

    train_dir = os.path.join(yolo_dir, 'train')

    yolo_home = YoloDatasetHome(yolo_dir)
    counter = 0
    grid_size = yolo_home.config['output_config']['grid_size']

    for x_batch, y_batch in yolo_home.flow_with_preload(train_dir, mini_batch_size=1):
        draw_boxes(x_batch[0], grid_size, y_batch[0])

        counter += 1

        if counter > 5:
            break
