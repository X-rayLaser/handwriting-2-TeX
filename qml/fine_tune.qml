import QtQuick 2.5
import QtQuick.Window 2.0
import QtQuick.Controls 2.2
import QtWebEngine 1.0

import "../qml/"

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

    readonly property var characters: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times'];

    Column {
        spacing: 10

        width: parent.width

        anchors.horizontalCenter: parent.horizontalCenter

        Text {
            id: predicted_expression

            anchors.horizontalCenter: parent.horizontalCenter
            text: "Draw a symbol '" + current_character + "'"
        }

        DrawingArea {
            id: drawing_area
            areaWidth: canvas_width
            areaHeight: canvas_height
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
                    drawing_area.erase();
                }
            }
            Button {
                id: add_button

                text: "Add example"

                onClicked: {
                    var res = drawing_area.canvasData()

                    manager.add_image(res.data, current_character, res.height, res.width);

                    drawing_area.erase();

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
