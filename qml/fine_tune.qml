import QtQuick 2.5
import QtQuick.Window 2.0
import QtQuick.Controls 2.2
import QtWebEngine 1.0

Window {
    id: root
    x: 400
    y: 400
    width: 300
    height: 300
    visible: true

    property int char_index: 0

    property string current_character: "0"

    property double val_accuracy: 0

    property int num_examples: 0

    property bool enough_examples: false

    readonly property int min_examples: 5

    readonly property int canvas_height: 90

    readonly property int canvas_width: 90

    Column {
        spacing: 10

        width: parent.width

        anchors.horizontalCenter: parent.horizontalCenter

        Text {
            id: predicted_expression

            anchors.horizontalCenter: parent.horizontalCenter
            text: "Draw a symbol '" + current_character + "'"
        }

        Rectangle {
            width: canvas_width
            height: canvas_height
            anchors.horizontalCenter: parent.horizontalCenter
            border.color: "red"

            Canvas {
                id: canvas
                width: parent.width
                height: parent.height

                onPaint: {
                    var ctx = getContext("2d");
                    ctx.fillStyle = "rgb(255, 255, 255)";
                    ctx.lineWidth = 3;
                    mouse_area.points.forEach(function (figurePoints) {
                        ctx.beginPath();
                        ctx.moveTo(figurePoints[0]);
                        ctx.lineTo(figurePoints[0]);
                        figurePoints.forEach(function (p) {
                            ctx.lineTo(p.x, p.y);
                            ctx.moveTo(p.x, p.y);
                        });
                    });
                    ctx.stroke();
                }

                MouseArea {
                    id: mouse_area
                    anchors.fill: parent
                    hoverEnabled: true

                    property var points : []

                    property bool pressed: false
                    onPressed: {
                        pressed = true;
                        points.push([]);
                    }

                    onReleased: {
                        pressed = false;
                    }
                    onPositionChanged: {
                        if (pressed === true) {
                            var figurePoints = points[points.length - 1];

                            figurePoints.push({x: mouseX, y: mouseY})

                            canvas.requestPaint();
                        }
                    }
                }
            }
        }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: "# of examples: " + String(num_examples)
        }

        Row {
            spacing: 10
            anchors.horizontalCenter: parent.horizontalCenter
            Button {
                text: "Erase"
                onClicked: {
                    mouse_area.points = [];
                    canvas.getContext("2d").reset();
                    canvas.requestPaint();
                }
            }
            Button {
                id: add_button

                text: "Add example"

                onClicked: {
                    var w = canvas.width;
                    var h = canvas.height;
                    var imageData = canvas.getContext("2d").getImageData(0, 0, w, h);
                    var data = [];
                    for (var i = 0; i < imageData.data.length; i++) {
                        data.push(imageData.data[i]);
                    }
                    var width = imageData.width;
                    var height = imageData.height;

                    manager.add_image(data, current_character, height, width);

                    mouse_area.points = [];
                    canvas.getContext("2d").reset();
                    canvas.requestPaint();

                    var characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times'];

                    char_index = char_index + 1;

                    if (char_index >= characters.length) {
                        char_index = 0;
                    }

                    current_character = characters[char_index];

                    num_examples = num_examples + 1;
                    if (num_examples > min_examples) {
                        calibrate_button.enabled = true;
                    }
                }
            }
        }

        Button {
            id: calibrate_button

            anchors.horizontalCenter: parent.horizontalCenter

            text: "Start tuning"

            enabled: false

            onClicked: {
                manager.fine_tune(30);
                calibrate_button.enabled = false;
                add_button.enabled = false;
                progress_bar.visible = true;
                calibration_info.visible = false;
            }
        }

        ProgressBar {
            id: progress_bar
            anchors.horizontalCenter: parent.horizontalCenter

            indeterminate: true
            visible: false
        }

        Text {
            id: calibration_info

            anchors.horizontalCenter: parent.horizontalCenter
            text: "Validation accuracy: " + String(val_accuracy.toFixed(2))

            visible: false
        }

        Connections {
            target: manager
            onTuningComplete: {
                val_accuracy = accuracy;
                add_button.enabled = true;
                num_examples = 0;
                progress_bar.visible = false;
                calibration_info.visible = true;
            }
        }

    }

}
