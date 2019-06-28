import QtQuick 2.5
import QtQuick.Controls 2.2


Rectangle {
    id: root
    property alias areaWidth: root.width
    property alias areaHeight: root.height
    border.color: "red"
    anchors.horizontalCenter: parent.horizontalCenter

    signal drawn(var data, int width, int height)

    function erase() {
        mouse_area.points = [];
        canvas.getContext("2d").reset();
        canvas.requestPaint();
    }

    function canvasData() {
        var w = canvas.width;
        var h = canvas.height;
        var imageData = canvas.getContext("2d").getImageData(0, 0, w, h);
        var data = [];
        for (var i = 0; i < imageData.data.length; i++) {
            data.push(imageData.data[i]);
        }
        var width = imageData.width;
        var height = imageData.height;
        return {
            data: data,
            width: width,
            height: height
        }
    }

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

                var res = canvasData();

                root.drawn(res.data, res.width, res.height);
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
