import threading
from json import loads, dumps

from websocket import WebSocketApp

SYMBOL = 'BTCUSDT'
INTERVAL1 = '1m'
INTERVAL2 = 'Min1'


class ExPriceDiff:
    def __init__(self, url1, url2) -> None:
        self.url1 = url1
        self.url2 = url2
        self.setup_websockets()

    def setup_websockets(self):
        self.ws1 = WebSocketApp(self.url1,
                                on_message=self.on_msg,
                                on_error=self.on_error,
                                on_close=self.on_close,
                                on_open=self.on_open)

        self.ws2 = WebSocketApp(self.url2,
                                on_message=self.on_msg,
                                on_error=self.on_error,
                                on_close=self.on_close,
                                on_open=self.on_open2)

    def on_msg(self, ws, message):
        data = loads(message)
        print(data)

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")

    def on_open(self, ws):
        print("### WebSocket 1 opened ###")

    def on_open2(self, ws):
        subscription_payload = {
            "method": "SUBSCRIPTION",
            "params": ["spot@public.kline.v3.api@BTCUSDT@Min1"]
        }
        ws.send(dumps(subscription_payload))
        print("### WebSocket 2 opened ###")

    def run_websocket(self, ws):
        ws.run_forever()

    def run(self):
        # Create two threads to run the WebSockets concurrently.
        threading.Thread(target=self.run_websocket, args=(self.ws1,)).start()
        threading.Thread(target=self.run_websocket, args=(self.ws2,)).start()


if __name__ == '__main__':
    url1 = f'wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_{INTERVAL1}'
    url2 = 'wss://wbs.mexc.com/ws'
    epd = ExPriceDiff(url1, url2)
    epd.run()
