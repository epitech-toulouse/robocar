#!/usr/bin/env bash
set -euo pipefail

safe_source() {
  local target_file="$1"
  local had_u=0
  case $- in
    *u*) had_u=1 ;;
  esac
  set +u
  # shellcheck disable=SC1090
  source "$target_file"
  if [[ $had_u -eq 1 ]]; then
    set -u
  fi
}

PKG_NAME="robocar_sim"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${ROBOCAR_WS:-$HOME/robocar_ws}"
MENU_MODE=false
MENU_TWO_TERMINALS=false
BUILD_ONLY=false
CLEAN_BUILD=false
START_RVIZ=false
LAUNCH_EXTRA_ARGS=()

pick_terminal_cmd() {
  if command -v gnome-terminal >/dev/null 2>&1; then
    echo "gnome-terminal"
    return 0
  fi
  if command -v konsole >/dev/null 2>&1; then
    echo "konsole"
    return 0
  fi
  if command -v xfce4-terminal >/dev/null 2>&1; then
    echo "xfce4-terminal"
    return 0
  fi
  if command -v xterm >/dev/null 2>&1; then
    echo "xterm"
    return 0
  fi
  return 1
}

open_terminal_window() {
  local term_bin="$1"
  local title="$2"
  local command_str="$3"

  case "$term_bin" in
    gnome-terminal)
      "$term_bin" --title="$title" -- bash -lc "$command_str; exec bash" &
      ;;
    konsole)
      "$term_bin" --new-tab -p tabtitle="$title" -e bash -lc "$command_str; exec bash" &
      ;;
    xfce4-terminal)
      "$term_bin" --title="$title" --command="bash -lc '$command_str; exec bash'" &
      ;;
    xterm)
      "$term_bin" -T "$title" -e bash -lc "$command_str; exec bash" &
      ;;
    *)
      return 1
      ;;
  esac
}

quote_args() {
  local out=""
  for arg in "$@"; do
    out+="$(printf '%q ' "$arg")"
  done
  printf '%s' "$out"
}

usage() {
  cat <<'EOF'
Usage: alexis.sh [options] [launch_args...]

Builds robocar_sim then runs it.

Options:
  -m, --menu               Start Gazebo/bridge + Qt menu only
  --menu-2term             With --menu, open Gazebo and Qt in separate terminals
  --rviz               Start RViz LiDAR window
  -w, --workspace <path>   ROS workspace path (default: ~/robocar_ws)
      --clean              Clean package build/install before building
      --build-only         Build only, do not run
  -h, --help               Show this help

Examples:
  ./alexis.sh
  ./alexis.sh --menu
  ./alexis.sh --menu --menu-2term
  ./alexis.sh --menu --menu-2term --rviz
  ./alexis.sh --menu --rviz
  ./alexis.sh --workspace ~/robocar_ws --menu
  ./alexis.sh start_controller:=false
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--menu)
      MENU_MODE=true
      shift
      ;;
    --menu-2term)
      MENU_MODE=true
      MENU_TWO_TERMINALS=true
      shift
      ;;
    --menu-qt)
      # Kept for backward compatibility; Qt mode is now default for --menu.
      MENU_MODE=true
      shift
      ;;
    -w|--workspace)
      if [[ $# -lt 2 ]]; then
        echo "Error: --workspace needs a path" >&2
        exit 1
      fi
      WS_DIR="$2"
      shift 2
      ;;
    --clean)
      CLEAN_BUILD=true
      shift
      ;;
    --rviz)
      START_RVIZ=true
      shift
      ;;
    --build-only)
      BUILD_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      LAUNCH_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$START_RVIZ" == true ]]; then
  LAUNCH_EXTRA_ARGS+=("start_rviz:=true")
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "Error: /opt/ros/jazzy/setup.bash not found. Install ROS 2 Jazzy first." >&2
  exit 1
fi

mkdir -p "$WS_DIR/src"
PKG_LINK="$WS_DIR/src/$PKG_NAME"

if [[ -L "$PKG_LINK" ]]; then
  LINK_TARGET="$(readlink -f "$PKG_LINK" || true)"
  if [[ "$LINK_TARGET" != "$SCRIPT_DIR" ]]; then
    rm -f "$PKG_LINK"
  fi
fi

if [[ -e "$PKG_LINK" && ! -L "$PKG_LINK" ]]; then
  echo "Error: $PKG_LINK exists and is not a symlink." >&2
  echo "Please remove or rename it, then re-run." >&2
  exit 1
fi

if [[ ! -e "$PKG_LINK" ]]; then
  ln -s "$SCRIPT_DIR" "$PKG_LINK"
fi

safe_source /opt/ros/jazzy/setup.bash

cd "$WS_DIR"

if [[ "$CLEAN_BUILD" == true ]]; then
  rm -rf "build/$PKG_NAME" "install/$PKG_NAME"
fi

colcon build --packages-select "$PKG_NAME" --symlink-install

safe_source "$WS_DIR/install/setup.bash"

if [[ "$BUILD_ONLY" == true ]]; then
  echo "Build done: $PKG_NAME"
  exit 0
fi

if [[ "$MENU_MODE" == true ]]; then
  if [[ "$MENU_TWO_TERMINALS" == true ]]; then
    if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
      echo "Warning: --menu-2term requested but no graphical display detected. Falling back to single-terminal menu mode."
      MENU_TWO_TERMINALS=false
    fi
  fi

  if [[ "$MENU_TWO_TERMINALS" == true ]]; then
    if ! TERM_BIN="$(pick_terminal_cmd)"; then
      echo "Warning: --menu-2term requested but no supported terminal emulator found (gnome-terminal/konsole/xfce4-terminal/xterm)."
      echo "Falling back to single-terminal menu mode."
      MENU_TWO_TERMINALS=false
    fi
  fi

  if [[ "$MENU_TWO_TERMINALS" == true ]]; then
    LAUNCH_ARGS=("start_controller:=true" "controller_menu_enabled:=false" "${LAUNCH_EXTRA_ARGS[@]}")
    LAUNCH_ARGS_QUOTED="$(quote_args "${LAUNCH_ARGS[@]}")"

    LAUNCH_CMD="source /opt/ros/jazzy/setup.bash && source $(printf '%q' "$WS_DIR")/install/setup.bash && ros2 launch $(printf '%q' "$PKG_NAME") robocar_sim.launch.py ${LAUNCH_ARGS_QUOTED}"
    QT_CMD="sleep 2 && source /opt/ros/jazzy/setup.bash && source $(printf '%q' "$WS_DIR")/install/setup.bash && ros2 run $(printf '%q' "$PKG_NAME") robocar_menu_qt"

    echo "Starting Gazebo/bridge in separate terminal..."
    open_terminal_window "$TERM_BIN" "Robocar Sim - Gazebo" "$LAUNCH_CMD"
    echo "Starting Qt control app in separate terminal..."
    open_terminal_window "$TERM_BIN" "Robocar Sim - Qt Menu" "$QT_CMD"
    echo "Two terminals launched (Gazebo + Qt Menu)."

    exit 0
  fi

  echo "Starting Gazebo/bridge/controller (controller menu disabled, Qt menu active)..."
  ros2 launch "$PKG_NAME" robocar_sim.launch.py start_controller:=true controller_menu_enabled:=false "${LAUNCH_EXTRA_ARGS[@]}" &
  LAUNCH_PID=$!

  cleanup() {
    if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
      kill "$LAUNCH_PID" 2>/dev/null || true
      wait "$LAUNCH_PID" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  sleep 2

  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    if TERM_BIN_QT="$(pick_terminal_cmd)"; then
      QT_CMD="source /opt/ros/jazzy/setup.bash && source $(printf '%q' "$WS_DIR")/install/setup.bash && ros2 run $(printf '%q' "$PKG_NAME") robocar_menu_qt"
      echo "Starting Qt control app in separate terminal..."
      open_terminal_window "$TERM_BIN_QT" "Robocar Sim - Qt Menu" "$QT_CMD"
      wait "$LAUNCH_PID"
    else
      echo "Warning: no supported terminal emulator found for Qt app. Run manually: ros2 run $PKG_NAME robocar_menu_qt"
      wait "$LAUNCH_PID"
    fi
  else
    echo "Warning: no graphical display detected for Qt app."
    wait "$LAUNCH_PID"
  fi
else
  ros2 launch "$PKG_NAME" robocar_sim.launch.py "${LAUNCH_EXTRA_ARGS[@]}"
fi
