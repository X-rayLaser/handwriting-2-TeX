import QtQuick 2.5
import QtQuick.Window 2.0
import QtQuick.Controls 2.2
import QtWebEngine 1.0

Window {
    id: root
    x: 400
    y: 400
    width: 400
    height: 400
    visible: true

    property string latex: "\\\\frac{1}{4}"

    Column {
        spacing: 10

        width: parent.width

        anchors.horizontalCenter: parent.horizontalCenter

        Canvas {
            id: canvas
            width: parent.width
            height: 200

            onPaint: {
                var ctx = getContext("2d");
                ctx.fillStyle = Qt.rgba(1, 0, 0, 1);
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
                    manager.recognize([]);
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

        WebEngineView {
            id: webEngineView
            width: parent.width
            height: 100
            url: "TeX_layout.html"

            onLoadProgressChanged: {
                if(loadProgress === 100){
                    webEngineView.runJavaScript("document.getElementById('latex_body').innerHTML = '$$" + latex + "$$';");
                }
            }
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
                text: "Eraze"
                onClicked: {
                    mouse_area.points = [];
                    canvas.getContext("2d").reset();
                    canvas.requestPaint();
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

        Connections {
            target: manager
            onPredictionReady: {
                latex = prediction;
                webEngineView.reload();
            }
        }

    }

}
