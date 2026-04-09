#!/bin/bash

oak_whisky_cmd_path() {
    printf '%s\n' "${OAK_WHISKY_CMD:-/Applications/Whisky.app/Contents/Resources/WhiskyCmd}"
}

oak_whisky_bottle_name() {
    printf '%s\n' "${OAK_WHISKY_BOTTLE:-CARLA}"
}

oak_whisky_prefix() {
    local whisky_cmd bottle shellenv prefix
    whisky_cmd="$(oak_whisky_cmd_path)"
    bottle="$(oak_whisky_bottle_name)"

    [ -x "$whisky_cmd" ] || return 1
    shellenv="$("$whisky_cmd" shellenv "$bottle")" || return 1
    prefix="$(printf '%s\n' "$shellenv" | awk -F '"' '/^export WINEPREFIX=/{ print $2; exit }')"
    [ -n "$prefix" ] || return 1

    printf '%s\n' "$prefix"
}

oak_whisky_python_host() {
    local prefix="${1:-}"
    local expected_version="${2:-}"
    local candidate

    [ -n "$prefix" ] || prefix="$(oak_whisky_prefix)" || return 1

    if [ -n "$expected_version" ]; then
        candidate="$(find "$prefix/drive_c" -type f -path "*Python${expected_version//./}*/python.exe" ! -path '*/Lib/venv/*' -print | sort | head -n 1)"
        [ -n "$candidate" ] || return 1
        printf '%s\n' "$candidate"
        return 0
    fi

    for candidate in \
        "$prefix/drive_c/users/crossover/AppData/Local/Programs/Python/Python310/python.exe" \
        "$prefix/drive_c/users/crossover/AppData/Local/Programs/Python/Python311/python.exe"
    do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    find "$prefix/drive_c" -type f -path '*/Python*/python.exe' -print | sort | head -n 1
}

oak_carla_egg_host() {
    local prefix="${1:-}"

    [ -n "$prefix" ] || prefix="$(oak_whisky_prefix)" || return 1

    find "$prefix/drive_c" -type f -path '*/PythonAPI/carla/dist/carla-*.egg' -print | head -n 1
}

oak_carla_app_host() {
    local prefix="${1:-}"
    local app_path

    [ -n "$prefix" ] || prefix="$(oak_whisky_prefix)" || return 1
    app_path="$prefix/drive_c/Program Files/CARLA_0.9.15/CARLA.app"
    [ -d "$app_path" ] || return 1

    printf '%s\n' "$app_path"
}

oak_carla_python_api_host() {
    local egg_path="${1:-}"

    [ -n "$egg_path" ] || egg_path="$(oak_carla_egg_host)" || return 1
    [ -n "$egg_path" ] || return 1

    dirname "$(dirname "$egg_path")"
}

oak_carla_python_version() {
    local egg_path="${1:-}"
    local egg_name

    [ -n "$egg_path" ] || egg_path="$(oak_carla_egg_host)" || return 1
    egg_name="$(basename "$egg_path")"

    if [[ "$egg_name" =~ -py([0-9]+)\.([0-9]+)-win-amd64\.egg$ ]]; then
        printf '%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return 0
    fi

    return 1
}

oak_carla_root_host() {
    local python_api_path="${1:-}"

    [ -n "$python_api_path" ] || python_api_path="$(oak_carla_python_api_host)" || return 1
    [ -n "$python_api_path" ] || return 1

    dirname "$(dirname "$python_api_path")"
}

oak_carla_exe_host() {
    local carla_root="${1:-}"

    [ -n "$carla_root" ] || carla_root="$(oak_carla_root_host)" || return 1
    [ -f "$carla_root/CarlaUE4.exe" ] || return 1

    printf '%s\n' "$carla_root/CarlaUE4.exe"
}

oak_to_windows_path() {
    local host_path="$1"
    local windows_path

    windows_path="${host_path//\//\\}"
    printf 'Z:%s\n' "$windows_path"
}

oak_export_carla_whisky_env() {
    local root_dir="$1"
    local prefix python_host egg_host python_api_host python_version carla_root_host carla_exe_host carla_app_host bridge_host

    prefix="$(oak_whisky_prefix)" || return 1
    egg_host="$(oak_carla_egg_host "$prefix")" || return 1
    python_version="$(oak_carla_python_version "$egg_host")" || return 1
    python_host="$(oak_whisky_python_host "$prefix" "$python_version")" || return 1
    python_api_host="$(oak_carla_python_api_host "$egg_host")" || return 1
    carla_root_host="$(oak_carla_root_host "$python_api_host")" || return 1
    carla_exe_host="$(oak_carla_exe_host "$carla_root_host")" || return 1
    carla_app_host="$(oak_carla_app_host "$prefix")" || return 1
    bridge_host="$root_dir/scripts/carla_whisky_bridge.py"

    [ -f "$python_host" ] || return 1
    [ -f "$egg_host" ] || return 1
    [ -d "$python_api_host" ] || return 1
    [ -f "$carla_exe_host" ] || return 1
    [ -d "$carla_app_host" ] || return 1
    [ -f "$bridge_host" ] || return 1

    export OAK_WHISKY_CMD="$(oak_whisky_cmd_path)"
    export OAK_WHISKY_BOTTLE="$(oak_whisky_bottle_name)"
    export OAK_CARLA_WIN_PYTHON="$(oak_to_windows_path "$python_host")"
    export OAK_CARLA_WIN_EGG="$(oak_to_windows_path "$egg_host")"
    export OAK_CARLA_WIN_PYTHONAPI="$(oak_to_windows_path "$python_api_host")"
    export OAK_CARLA_WIN_EXE="$(oak_to_windows_path "$carla_exe_host")"
    export OAK_CARLA_APP_HOST="$carla_app_host"
    export OAK_CARLA_PYTHON_VERSION="$python_version"
    export OAK_CARLA_BRIDGE=1
}
