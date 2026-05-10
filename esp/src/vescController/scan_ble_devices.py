#!/usr/bin/env python3

import argparse
import asyncio
from typing import Iterable

from bleak import BleakScanner


ROBOCAR_CAMERA_SERVICE_UUID = "0100aaaf-6d66-7b98-2f4d-60a8c0265631"


def format_hex(data: bytes) -> str:
    return data.hex(" ") if data else "-"


def format_uuid_list(uuids: Iterable[str]) -> str:
    values = list(uuids)
    if not values:
        return "-"
    return ", ".join(values)


def format_manufacturer_data(manufacturer_data: dict[int, bytes]) -> str:
    if not manufacturer_data:
        return "-"
    lines = []
    for company_id, payload in sorted(manufacturer_data.items()):
        lines.append(f"0x{company_id:04x}: {format_hex(payload)}")
    return "\n".join(lines)


def format_service_data(service_data: dict[str, bytes | str]) -> str:
    if not service_data:
        return "-"
    lines = []
    for uuid, payload in sorted(service_data.items()):
        if isinstance(payload, bytes):
            rendered = format_hex(payload)
        else:
            rendered = str(payload)
        lines.append(f"{uuid}: {rendered}")
    return "\n".join(lines)


def looks_like_robocar(metadata_uuids: list[str], local_name: str | None) -> bool:
    if local_name == "robocar-camera":
        return True
    return any(uuid.lower() == ROBOCAR_CAMERA_SERVICE_UUID for uuid in metadata_uuids)


def adv_attr(advertisement_data, name: str, default="-"):
    return getattr(advertisement_data, name, default)


async def scan(duration: float) -> None:
    print(f"Scanning BLE for {duration:.1f}s...\n")
    devices = await BleakScanner.discover(timeout=duration, return_adv=True)
    if not devices:
        print("No BLE devices found.")
        return

    sorted_devices = sorted(
        devices.values(),
        key=lambda item: (
            item[0].name or item[1].local_name or "",
            item[0].address,
        ),
    )

    for device, adv in sorted_devices:
        name = device.name or adv.local_name or "-"
        uuids = adv.service_uuids or []
        marker = "  <== likely ESP robocar" if looks_like_robocar(uuids, adv.local_name or device.name) else ""

        print(f"Device: {name}{marker}")
        print(f"  Address: {device.address}")
        print(f"  RSSI: {adv.rssi}")
        print(f"  Local name: {adv.local_name or '-'}")
        print(f"  Connectable: {adv_attr(adv, 'connectable')}")
        tx_power = adv_attr(adv, "tx_power", None)
        print(f"  TX power: {tx_power if tx_power is not None else '-'}")
        print(f"  Service UUIDs: {format_uuid_list(uuids)}")
        print(f"  Manufacturer data:\n{indent_block(format_manufacturer_data(adv.manufacturer_data), 4)}")
        print(f"  Service data:\n{indent_block(format_service_data(adv.service_data), 4)}")
        print()


def indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan BLE devices and print useful identification data."
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=8.0,
        help="scan duration in seconds (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(scan(args.time))


if __name__ == "__main__":
    main()
