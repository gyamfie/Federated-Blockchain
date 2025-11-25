import socket
import time
import psutil
import struct
import textwrap
from ina219 import INA219
from ina219 import DeviceRangeError
import csv

#TAB_1 ='\t - '
#TAB_2 ='\t\t - '
#TAB_3 ='\t\t\t - '
#TAB_4 ='\t\t\t\t - '

#DATA_TAB_1 = '\t - '
#DATA_TAB_2 = '\t\t - '
#DATA_TAB_3 = '\t\t\t - '
#DATA_TAB_4 = '\t\t\t\t - '

def main():
    conn= socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))

    while True:

        raw_data, addr = conn.recvfrom(65536)
        dest_mac, src_mac, eth_proto, data =ethernet_frame(raw_data)
        power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage = read()
        #print('\nEthernet Frame: ')
        #print(TAB_1 + 'Destination: {}, Source: {}, Protocol:{}'.format(dest_mac, src_mac, eth_proto))

        # 8 for IPv4
        if eth_proto == 8:
            (version, header_length,ttl, proto, src, target, data) = ipv4_packet(data)
            # ICMP
            if proto == 1:
                (icmp_type, code, checksum, data) = icmp_packet(data)
                dataframe = [power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage, dest_mac, src_mac, eth_proto, version, header_length,ttl, proto, src, target, icmp_type, code, checksum]
                # open the file in the write mode
                f = open('csv_icmp', 'w')
                writer = csv.writer(f)
                writer.writerow(dataframe)
                f.close()
                print(power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage, dest_mac, src_mac, eth_proto, version, header_length,ttl, proto, src, target, icmp_type, code, checksum)
            # TCP
            elif proto == 6:
                (src_port, dest_port, sequence, acknowledgement, flag_urg, flag_ack, flag_psh, flag_rst, flag_syn, flag_fin, data) = tcp_segment(data)
                dataframe = [power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage, dest_mac, src_mac, eth_proto,version, header_length,ttl, proto, src, target, src_port, dest_port, sequence, acknowledgement, flag_urg, flag_ack, flag_psh, flag_rst, flag_syn, flag_fin]
                f = open('csv_tcp', 'w')
                writer = csv.writer(f)
                writer.writerow(dataframe)
                f.close()
                print(power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage, dest_mac, src_mac, eth_proto,version, header_length,ttl, proto, src, target, src_port, dest_port, sequence, acknowledgement, flag_urg, flag_ack, flag_psh, flag_rst, flag_syn, flag_fin)
            #  UDP
            elif proto == 17:
                src_port, dest_port, size, data = udp_segment(data)
                dataframe = [power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage, dest_mac, src_mac, eth_proto,version, header_length,ttl, proto, src, target,src_port, dest_port, size]
                f = open('csv_udp', 'w')
                writer = csv.writer(f)
                writer.writerow(dataframe)
                f.close()
                print(power, byte_sent, byte_in, packet_sent, packet_in, cpu_usage, dest_mac, src_mac, eth_proto,version, header_length,ttl, proto, src, target,src_port, dest_port, size )

        time.sleep(2)

def ipv4_packet(data):
    version_header_length = data[0]
    version = version_header_length >> 4
    header_length= (version_header_length & 15) * 4
    ttl, proto, src, target = struct.unpack('! 8x B B 2x 4s 4s', data[:20])
    return version, header_length,ttl, proto,ipv4(src), ipv4(target), data[header_length:]

#  Return properly formatted IPV4 address
def ipv4(addr):
    return '.'.join(map(str, addr))

# ICMP packet
def icmp_packet(data):
    icmp_type, code, checksum = struct.unpack('! B B H', data[:4])
    return icmp_type, code, checksum, data[4:]

# TCP segment
def tcp_segment(data):
    (src_port, dest_port, sequence, acknowledgement, offset_reserved_flags)= struct.unpack('! H H L L H', data[:14])
    offset = (offset_reserved_flags >> 12) * 4
    flag_urg = (offset_reserved_flags & 32) >> 5
    flag_ack = (offset_reserved_flags & 16) >> 4
    flag_psh = (offset_reserved_flags & 8) >> 3
    flag_rst = (offset_reserved_flags & 4) >> 2
    flag_syn = (offset_reserved_flags & 2) >> 1
    flag_fin = offset_reserved_flags & 1
    return src_port, dest_port, sequence, acknowledgement, flag_urg, flag_ack, flag_psh, flag_rst, flag_syn, flag_fin, data[offset:]

# UDP segment
def udp_segment(data):
    src_port, dest_port, size = struct.unpack('! H H 2x H', data[:8])
    return src_port, dest_port, size, data[8:]

def format_multi_line(prefix, string, size=80):
    size -= len(prefix)
    if isinstance(string, bytes):
        string =''.join(r'\x{:02x}'.format(byte) for byte in string)
        if size % 2:
            size -= 1
    return '\n'.join([prefix + line for line in textwrap.wrap(string, size)])



def ethernet_frame(data):
    dest_mac,src_mac, proto = struct.unpack('! 6s 6s H', data[:14])
    return get_mac_addr(dest_mac), get_mac_addr(src_mac), socket.htons(proto), data[14:]

#Return formatted MAC
def get_mac_addr(bytes_addr):
    bytes_addr= map('{:02x}'.format, bytes_addr)
    return ':'.join(bytes_addr).upper()

def read():
    ina = INA219(SHUNT_OHMS)
    ina.configure()
    cpu_usage = psutil.cpu_percent()
    network_info = psutil.net_io_counters()
    return ina.power(), network_info.bytes_sent, network_info.bytes_recv, network_info.packets_sent, network_info.packets_recv, cpu_usage
    # print("CPU Usage: ", cpu_usage)
    # print(network_info)
    #print("Bus Voltage: %.3f V" % ina.voltage())
   # try:
   #     print("Bus Current: %.3f mA" % ina.current())
    #    print("Power: %.3f mW" % ina.power())
    #    print("Shunt voltage: %.3f mV" % ina.shunt_voltage())

   # except DeviceRangeError as e:
        # Current out of device range with specified shunt resistor
    #    print(e)

if __name__ == "__main__":
    main()
