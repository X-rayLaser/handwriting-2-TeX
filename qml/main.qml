import QtQuick 2.5
import QtQuick.Window 2.0
import QtQuick.Controls 2.2
import QtWebEngine 1.0

import "../qml/"

Window {
    id: root
    x: 400
    y: 400
    width: 400
    height: 700
    visible: true

    color: "grey"

    property string latex: ""
    property string escaped_latex: ""

    property string current_classifier: "classification_model.h5"

    readonly property int canvas_width: 400
    readonly property int canvas_height: 300

    Column {
        spacing: 10

        width: parent.width

        anchors.horizontalCenter: parent.horizontalCenter

        Column {
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 2

            Repeater {
                model: manager.classifiers

                Rectangle {
                    width: 200
                    height: 20
                    radius: 3
                    Text {
                        width: parent.width
                        height: parent.height
                        anchors.horizontalCenter: parent.horizontalCenter

                        text: modelData

                        MouseArea {
                            anchors.fill: parent

                            onClicked: {
                                current_classifier = modelData;
                                manager.set_classifier(current_classifier);
                            }
                        }
                    }
                }
            }
        }

        Text {
            id: current_model_text

            anchors.horizontalCenter: parent.horizontalCenter
            text: "Current model: " + current_classifier
        }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: "Drawing area"
        }

        DrawingArea {
            id: drawing_area
            areaWidth: canvas_width
            areaHeight: canvas_height

            onDrawn: {
                manager.recognize(data, width, height);
                progress_bar.visible = true;
            }
        }

        WebEngineView {
            id: webEngineView
            width: parent.width
            height: 100
            url: "TeX_layout.html"

            onLoadProgressChanged: {
                if(loadProgress === 100){
                    webEngineView.runJavaScript("document.getElementById('latex_body').innerHTML = '$$" + escaped_latex + "$$';");
                }
            }
        }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: "TeX markup"
        }
        Text {
            id: predicted_expression

            anchors.horizontalCenter: parent.horizontalCenter
            text: latex
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
                id: download_button

                text: "Copy to clipboard"

                onClicked: {
                    manager.copy_to_clipboard(latex);
                }
            }
        }

        ProgressBar {
            id: progress_bar
            anchors.horizontalCenter: parent.horizontalCenter

            indeterminate: true
            visible: false
        }

        Connections {
            target: manager
            onPredictionReady: {
                latex = prediction;
                escaped_latex = latex.split("\\").join("\\\\");
                webEngineView.reload();
                progress_bar.visible = manager.in_progress;
            }
        }

    }

}
