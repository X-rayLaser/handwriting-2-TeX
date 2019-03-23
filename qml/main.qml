import QtQuick 2.5
import QtQuick.Window 2.0
import QtQuick.Controls 2.2
import QtQuick.Dialogs 1.1


Window {
    id: root
    x: 400
    y: 400
    width: 600
    height: 650
    visible: true

    Column {
        spacing: 10

        width: parent.width

        anchors.horizontalCenter: parent.horizontalCenter

        Canvas {
            id: canvas
            width: parent.width
            height: 400

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

        Text {
            text: "\\sum{i=0}^{N}"
        }

        Button {
            id: download_button
            text: "Copy to clipboard"
        }

    }

}
