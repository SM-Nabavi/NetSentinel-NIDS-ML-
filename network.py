import time
from scapy.all import IP, TCP, UDP
import numpy as np
from collections import deque


MODEL_FEATURES = [
    "Protocol",
    "FlowDuration",
    "TotalPackets",
    "TotalBytes",
    "ForwardPackets",
    "BackwardPackets",
    "ForwardBytes",
    "BackwardBytes",
    "PacketLengthMin",
    "PacketLengthMax",
    "PacketLengthMean",
    "PacketLengthStd",
    "PacketLengthVariance",
    "FlowIATMean",
    "FlowIATStd",
    "FlowIATMin",
    "FlowIATMax",
    "FlowPacketsPerSecond",
    "FlowBytesPerSecond",
    "AveragePacketSize",
    "BytesPerPacket",
    "SYNFlagCount",
    "ACKFlagCount",
    "PSHFlagCount",
    "RSTFlagCount",
    "FINFlagCount",
    "URGFlagCount",
    "SynAckRatio",
    "ForwardBackwardPacketRatio",
    "ForwardBackwardByteRatio",
    "PacketSizeEntropy"
]

class Flow:
    def __init__(self,key):
        self.key = key
        # ---------- زمان‌ها ----------
        self.first_packet_time = None
        self.last_packet_time = None
        self.last_seen = time.time()

        # ---------- اطلاعات پایه (از اولین بسته) ----------
        self.src_ip = None
        self.dst_ip = None
        self.protocol = None
        self.src_port = 0
        self.dst_port = 0

        # ---------- شمارنده‌های تجمعی ----------
        self.total_packets = 0
        self.total_bytes = 0
        self.forward_packets = 0     
        self.backward_packets = 0    
        self.forward_bytes = 0
        self.backward_bytes = 0

        # ---------- شمارنده پرچم‌های TCP ----------
        self.syn_count = 0
        self.ack_count = 0
        self.psh_count = 0
        self.rst_count = 0
        self.fin_count = 0
        self.urg_count = 0

        self.packet_lengths = deque(maxlen=200)
        self.timestamps = deque(maxlen=200)
    def add_packet(self,pkt):
        now = time.time()
        self.last_seen = now
        pkt_len = len(pkt)

        self.total_packets += 1 
        self.total_bytes += pkt_len
        #ثبت زمان و اطلاعات
        if self.first_packet_time is None:
            self.first_packet_time = float(pkt(time))
            self.src_ip = pkt[IP].src
            self.dst_ip = pkt[IP].dst
            self.protocol = pkt[IP].proto

        #تشخیص پورت
        if TCP in pkt:
            self.src_port = pkt[TCP].sport
            self.dst_port = pkt[TCP].dport
        elif UDP in pkt:
            self.src_port = pkt[UDP].sport
            self.dst_port = pkt[UDP].dport
        
        
        self.last_packet_time = float(pkt.time)

        #تشخیص جهت
        if pkt[IP].src == self.src_ip:
            self.forward_packets += 1
            self.forward_bytes += pkt_len
        else:
            self.backward_packets += 1
            self.backward_bytes += pkt_len

        # شمارش پرچم‌های TCP
        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            if flags & 0x02: self.syn_count += 1
            if flags & 0x10: self.ack_count += 1
            if flags & 0x08: self.psh_count += 1
            if flags & 0x04: self.rst_count += 1
            if flags & 0x01: self.fin_count += 1
            if flags & 0x20: self.urg_count += 1

        self.packet_lengths.append(pkt_len)
        self.timestamps.append(float(pkt.time))
    def get_feature_dict(self):
        if self.first_packet_time is None or self.total_packets < 2:
            return None
        
        duration = self.last_packet_time - self.first_packet_time
        lengths = list(self.packet_lengths)
        times = list(self.timestamps)
        iats = []
        if len(times) > 1:
            iats = np.diff(times)

        features = {
            "Protocol": self.protocol,
            "FlowDuration": duration,
            "TotalPackets": self.total_packets,
            "TotalBytes": self.total_bytes,
            "ForwardPackets": self.forward_packets,
            "BackwardPackets": self.backward_packets,
            "ForwardBytes": self.forward_bytes,
            "BackwardBytes": self.backward_bytes,

            "PacketLengthMin": float(np.min(lengths)),
            "PacketLengthMax": float(np.max(lengths)),
            "PacketLengthMean": float(np.mean(lengths)),
            "PacketLengthStd": float(np.std(lengths)),
            "PacketLengthVariance": float(np.var(lengths)),

            "FlowIATMean": float(np.mean(iats)) if len(iats) > 0 else 0.0,
            "FlowIATStd": float(np.std(iats)) if len(iats) > 0 else 0.0,
            "FlowIATMin": float(np.min(iats)) if len(iats) > 0 else 0.0,
            "FlowIATMax": float(np.max(iats)) if len(iats) > 0 else 0.0,

            "FlowPacketsPerSecond": self.total_packets / duration,
            "FlowBytesPerSecond": self.total_bytes / duration,
            "AveragePacketSize": self.total_bytes / self.total_packets,
            "BytesPerPacket": self.total_bytes / self.total_packets,  # مشابه بالا

            "SYNFlagCount": self.syn_count,
            "ACKFlagCount": self.ack_count,
            "PSHFlagCount": self.psh_count,
            "RSTFlagCount": self.rst_count,
            "FINFlagCount": self.fin_count,
            "URGFlagCount": self.urg_count,

            "SynAckRatio": self.syn_count / self.ack_count if self.ack_count > 0 else min(self.syn_count, 1000),
            "ForwardBackwardPacketRatio": self.forward_packets / self.backward_packets if self.backward_packets > 0 else min(self.forward_packets, 1000),
            "ForwardBackwardByteRatio": self.forward_bytes / self.backward_bytes if self.backward_bytes > 0 else min(self.forward_bytes, 1000),

            "PacketSizeEntropy": self._calculate_entropy(lengths)
        }
        return features
    
    @staticmethod
    def _calculate_entropy(values):
        """محاسبه آنتروپی شانون برای لیست اعداد"""
        if not values:
            return 0.0
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / counts.sum()
        return -float(np.sum(probs * np.log2(probs)))
    
class IDS_ML:
    def __init__(self):
        self.flows = {}
        self.packet_counter = 0
        self.ML_detector = HybridMLDetector()
        self.IDS_engine = SignatureEngine()
        self.alert_manager = AlertManager()
        self.logger = LoggerManager()
        self.queue_manager = QueueManager()
    def generate_flow_key(self,pkt):
        if IP not in pkt:
            return None
        ip = pkt[IP]

        src = ip.src
        dst = ip.dst

        proto = ip.proto

        sport = 0
        dport = 0

        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport

        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport

        endpoints = sorted([
            (src, sport),
            (dst, dport)
        ])

        return (
            endpoints[0],
            endpoints[1],
            proto
        )

