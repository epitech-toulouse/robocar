#!/usr/bin/env python3

import sys
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt5 is required. Install with: sudo apt install python3-pyqt5")
    raise


class RobocarMenuNode(Node):
    def __init__(self) -> None:
        super().__init__("robocar_menu_qt")
        self.mode_pub = self.create_publisher(Bool, "/robocar/menu/autonomous_enabled", 10)
        self.manual_pub = self.create_publisher(Twist, "/robocar/menu/manual_cmd_vel", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def set_mode(self, autonomous: bool) -> None:
        msg = Bool()
        msg.data = autonomous
        self.mode_pub.publish(msg)

    def send_manual(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.manual_pub.publish(msg)
        self.cmd_vel_pub.publish(msg)


class RobocarMenuWindow(QWidget):
    def __init__(self, node: RobocarMenuNode) -> None:
        super().__init__()
        self.node = node
        self.stream_enabled = False

        self.setWindowTitle("Robocar Menu (Qt)")
        self.resize(420, 260)

        self.mode_label = QLabel("Mode: unknown")

        mode_box = QGroupBox("Mode")
        mode_layout = QGridLayout()
        self.btn_auto = QPushButton("AUTO")
        self.btn_manual = QPushButton("MANUAL")
        self.btn_stop = QPushButton("STOP")
        mode_layout.addWidget(self.btn_auto, 0, 0)
        mode_layout.addWidget(self.btn_manual, 0, 1)
        mode_layout.addWidget(self.btn_stop, 0, 2)
        mode_box.setLayout(mode_layout)

        cmd_box = QGroupBox("Manual cmd_vel")
        cmd_layout = QGridLayout()

        self.linear_spin = QDoubleSpinBox()
        self.linear_spin.setRange(-3.0, 3.0)
        self.linear_spin.setSingleStep(0.1)
        self.linear_spin.setValue(0.8)

        self.angular_spin = QDoubleSpinBox()
        self.angular_spin.setRange(-3.0, 3.0)
        self.angular_spin.setSingleStep(0.1)
        self.angular_spin.setValue(0.0)

        self.btn_send_once = QPushButton("Send once")
        self.btn_stream_toggle = QPushButton("Start stream (10 Hz)")

        cmd_layout.addWidget(QLabel("Linear x (m/s)"), 0, 0)
        cmd_layout.addWidget(self.linear_spin, 0, 1)
        cmd_layout.addWidget(QLabel("Angular z (rad/s)"), 1, 0)
        cmd_layout.addWidget(self.angular_spin, 1, 1)
        cmd_layout.addWidget(self.btn_send_once, 2, 0)
        cmd_layout.addWidget(self.btn_stream_toggle, 2, 1)
        cmd_box.setLayout(cmd_layout)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.mode_label)
        root_layout.addWidget(mode_box)
        root_layout.addWidget(cmd_box)
        self.setLayout(root_layout)

        self.btn_auto.clicked.connect(self.on_auto_clicked)
        self.btn_manual.clicked.connect(self.on_manual_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_send_once.clicked.connect(self.on_send_once_clicked)
        self.btn_stream_toggle.clicked.connect(self.on_stream_toggled)

        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self.publish_manual_cmd)
        self.stream_timer.setInterval(100)

    def on_auto_clicked(self) -> None:
        if self.stream_enabled:
            self.on_stream_toggled()
        self.node.set_mode(True)
        self.mode_label.setText("Mode: AUTO")

    def on_manual_clicked(self) -> None:
        self.node.set_mode(False)
        self.mode_label.setText("Mode: MANUAL (streaming)")
        if not self.stream_enabled:
            self.on_stream_toggled()

    def on_stop_clicked(self) -> None:
        if self.stream_enabled:
            self.on_stream_toggled()
        self.node.set_mode(False)
        self.node.send_manual(0.0, 0.0)
        self.mode_label.setText("Mode: MANUAL (stopped)")

    def publish_manual_cmd(self) -> None:
        self.node.send_manual(float(self.linear_spin.value()), float(self.angular_spin.value()))

    def on_send_once_clicked(self) -> None:
        self.publish_manual_cmd()

    def on_stream_toggled(self) -> None:
        self.stream_enabled = not self.stream_enabled
        if self.stream_enabled:
            self.stream_timer.start()
            self.btn_stream_toggle.setText("Stop stream")
        else:
            self.stream_timer.stop()
            self.btn_stream_toggle.setText("Start stream (10 Hz)")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stream_timer.stop()
        super().closeEvent(event)


def main(argv: Optional[list[str]] = None) -> int:
    app = QApplication(sys.argv)

    rclpy.init(args=argv)
    node = RobocarMenuNode()

    window = RobocarMenuWindow(node)
    window.show()

    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
