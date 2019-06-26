import QtQuick 2.5
import QtQuick.Window 2.0
import QtQuick.Controls 2.2
import QtWebEngine 1.0

Window {
    id: root
    x: 400
    y: 400
    width: 400
    height: 650
    visible: true

    property int char_index: 0

    property string current_character: "0"

    property double val_accuracy: 0

    Column {
        spacing: 10

        width: parent.width

        anchors.horizontalCenter: parent.horizontalCenter

        Text {
            id: predicted_expression

            anchors.horizontalCenter: parent.horizontalCenter
            text: "Draw a symbol " + current_character
        }

        Rectangle {
            width: 90
            height: 90
            anchors.horizontalCenter: parent.horizontalCenter
            border.color: "red"

            Canvas {
                id: canvas
                width: 90
                height: 90

                onPaint: {
                    var ctx = getContext("2d");
                    ctx.fillStyle = "rgb(255, 255, 255)";
                    mouse_area.points.forEach(function (figurePoints) {
                        ctx.beginPath();
                        ctx.moveTo(figurePoints[0]);
                        ctx.lineTo(figurePoints[0]);
                        figurePoints.forEach(function (p) {
                            ctx.lineTo(p.x, p.y)
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

        Row {
            spacing: 10
            anchors.horizontalCenter: parent.horizontalCenter
            Button {
                text: "Eraze"
                onClicked: {
                    mouse_area.points = [];
                    canvas.getContext("2d").reset();
                    canvas.requestPaint();
                }
            }
            Button {
                id: download_button

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

                    manager.add_image(data, current_character);

                    mouse_area.points = [];
                    canvas.getContext("2d").reset();
                    canvas.requestPaint();

                    var characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times'];

                    char_index = char_index + 1;

                    if (char_index >= characters.length) {
                        char_index = 0;
                    }

                    current_character = characters[char_index];
                }
            }
        }

        Button {
            id: calibrate_button

            text: "Start tuning"

            onClicked: {
                manager.fine_tune();
                calibrate_button.enabled = false;
            }
        }

        Text {
            id: calibration_info

            anchors.horizontalCenter: parent.horizontalCenter
            text: "Validation accuracy: " + String(val_accuracy)
        }

        Connections {
            target: manager
            onTuningComplete: {
                val_accuracy = accuracy;
                calibrate_button.enabled = true;
            }
        }

    }

}
