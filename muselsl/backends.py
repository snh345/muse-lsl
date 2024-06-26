import asyncio
import atexit
import time
try:
    import bleak
except ModuleNotFoundError as error:
    bleak = error

def _wait(coroutine):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)

def sleep(seconds):
    time.sleep(seconds)

class BleakBackend:
    def __init__(self):
        self.connected = set()
        atexit.register(self.stop)
        # run the event loop when sleeping
        global sleep
        sleep = self.pump
    def start(self):
        pass
    def pump(self, seconds=1):
        _wait(asyncio.sleep(seconds))
    def stop(self):
        for device in [*self.connected]:
            device.disconnect()
    def scan(self, timeout=10):
        if isinstance(bleak, ModuleNotFoundError):
            raise bleak
        devices = _wait(bleak.BleakScanner.discover(timeout))
        return [{'name':device.name, 'address':device.address} for device in devices]
    def connect(self, address, connection_timeout=None):
        result = BleakDevice(self, address, connection_timeout)
        result.connect()
        return result

class BleakDevice:
    def __init__(self, adapter, address, connection_timeout=None):
        self._adapter = adapter
        self._timeout = connection_timeout
        self._client = bleak.BleakClient(address, timeout=self._timeout)  # <- CRITICAL timeout value HERE
    def connect(self):
        _wait(self._client.connect())
        self._adapter.connected.add(self)
    def disconnect(self):
        _wait(self._client.disconnect())
        self._adapter.connected.remove(self)
    # Characteristics have two handles: the declaration handle and the value handle.
    # Pygatt seems to use the value handle, which appears less common.  Bleak uses the
    # declaration handle used by d-bus.
    # With the muse, the declaration and value handles happen to be sequential.
    # So, we subtract 1 to get the declaration handle, and add 1 to get the value handle.
    def char_write_handle(self, value_handle, value, wait_for_response=True, timeout=30):
        declaration_handle = value_handle - 1
        _wait(self._client.write_gatt_char(
            declaration_handle,
            bytearray(value),
            wait_for_response))
    def subscribe(self, uuid, callback=None, indication=False, wait_for_response=True):
        def wrap(gatt_characteristic, data):
            value_handle = gatt_characteristic.handle + 1
            callback(value_handle, data)
        _wait(self._client.start_notify(uuid, wrap))
