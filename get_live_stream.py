import asyncio
import threading
import logging
import cv2
import queue
import time
import aiohttp
import sys
from aiortc import RTCPeerConnection, RTCSessionDescription

# 屏蔽繁杂的日志
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)


class WebRTCStreamer:
    def __init__(self, url):
        self.url = url
        self.frame_queue = queue.Queue(maxsize=2)  # 只保留最新的2帧，防止积压导致延迟
        self.running = False
        self.thread = None
        self.pc = None

    def start(self):
        """启动后台线程拉流"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print("✅ [视频流] 后台线程已启动，正在连接 WebRTC...")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_latest_frame(self):
        """
        获取最新的一帧图像 (OpenCV/Numpy format)
        如果队列为空，返回 None
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _run_event_loop(self):
        # Windows下必须设置这个策略
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.pc = RTCPeerConnection()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.ensure_future(self._consume_track(track))

        try:
            loop.run_until_complete(self._connect_whep())
            # 保持 loop 运行，直到 stop 被调用
            while self.running:
                loop.run_until_complete(asyncio.sleep(0.1))
        except Exception as e:
            print(f"❌ [视频流] 循环错误: {e}")
        finally:
            loop.run_until_complete(self.pc.close())
            loop.close()

    async def _connect_whep(self):
        self.pc.addTransceiver("video", direction="recvonly")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.url,
                    data=self.pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"}
            ) as response:
                if response.status not in [200, 201]:
                    print(f"❌ [视频流] 服务器连接失败: {response.status}")
                    return
                answer_sdp = await response.text()

        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self.pc.setRemoteDescription(answer)
        print("✅ [视频流] WebRTC 连接成功！")

    async def _consume_track(self, track):
        while self.running:
            try:
                frame = await track.recv()
                # 转换为 OpenCV BGR 格式
                img = frame.to_ndarray(format="bgr24")

                # 放入队列，如果满了就丢弃旧的，保证实时性
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(img)
            except Exception:
                pass


# 测试代码
if __name__ == "__main__":
    streamer = WebRTCStreamer("http://101.132.172.117:8889/live/psdk-client-M350/whep")
    streamer.start()
    while True:
        frame = streamer.get_latest_frame()
        if frame is not None:
            cv2.imshow("Test Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    streamer.stop()