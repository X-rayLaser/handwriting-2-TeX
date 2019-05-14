def draw_boxes(a, grid_size, output_volume):
    from PIL.ImageDraw import ImageDraw
    from data_synthesis import array_to_image

    height, width = a.shape
    cell_height = height / grid_size
    cell_width = width / grid_size
    print(height, width, cell_height, cell_width, output_volume.shape)

    image = array_to_image(a)

    canvas = ImageDraw(image)

    for row in range(grid_size):
        for col in range(grid_size):
            if output_volume[row, col, 0] > 0.6:
                print(output_volume[row, col])
                cell_x = col * cell_width
                cell_y = row * cell_height
                box = output_volume[row, col][1:5]
                xc, yc, w, h = box
                print(xc, yc, w, h)

                xc_abs = cell_x + xc * cell_width
                yc_abs = cell_y + yc * cell_height

                w_abs = int(round(w * cell_width))
                h_abs = int(round(h * cell_height))

                x = int(round(xc_abs - w_abs / 2))
                y = int(round(yc_abs - h_abs / 2))

                print(x, y, w_abs, h_abs)

                xy = [(x, y), (x + w_abs, y + h_abs)]
                canvas.rectangle(xy, width=4, outline=128)

    image.show()


if __name__ == '__main__':
    import os
    from yolo.yolo_home import YoloDatasetHome

    yolo_dir = '../datasets/yolo_dataset'

    train_dir = os.path.join(yolo_dir, 'train')

    yolo_home = YoloDatasetHome(yolo_dir)
    counter = 0
    for x_batch, y_batch in yolo_home.flow_with_preload(train_dir, mini_batch_size=1):
        draw_boxes(x_batch[0], 9, y_batch[0])

        counter += 1

        if counter > 5:
            break
