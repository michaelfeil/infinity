import zmq
import multiprocessing
import time
import psutil  # For memory checks

def get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5 GB
    else:
        buf_size = -1  # Use system default

    socket = context.socket(socket_type)
    if socket_type == zmq.PUB:
        socket.setsockopt(zmq.SNDHWM, 0)  # No limit on messages queued
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.bind(endpoint)
    elif socket_type == zmq.SUB:
        socket.setsockopt(zmq.RCVHWM, 0)  # No limit on messages queued
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.connect(endpoint)
    elif socket_type == zmq.REP:
        socket.bind(endpoint)
    elif socket_type == zmq.REQ:
        socket.connect(endpoint)
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    return socket

def subscriber_process(ident):
    context = zmq.Context()
    # Synchronization socket to signal readiness
    sync_socket = get_zmq_socket(context, zmq.REQ, "ipc://sync.ipc")
    
    # Subscriber socket
    socket = get_zmq_socket(context, zmq.SUB, "ipc://pubsub.ipc")
    # Subscribe to messages with the given ident as bytes
    topic_filter = ident.to_bytes(1, byteorder='big')
    socket.setsockopt(zmq.SUBSCRIBE, topic_filter)
    
    # Signal readiness to publisher
    sync_socket.send(b'READY')
    sync_socket.recv()  # Wait for acknowledgment
    
    running = True
    while running:
        try:
            # Receive multipart message: [topic][payload]
            topic = socket.recv()
            payload = socket.recv_pyobj()
            topic_int = int.from_bytes(topic, 'big')
            print(f"Subscriber {ident} received on topic {topic_int}: {payload}")
            # Check for stop command
            if payload.get('command') == 'STOP':
                running = False
        except Exception as e:
            print(f"Subscriber {ident} exception: {e}")
            running = False
    socket.close()
    context.term()

def spawn_one_subscriber(ident):
    # Create and start a subscriber process
    p = multiprocessing.Process(target=subscriber_process, args=(ident,))
    p.start()
    return p

def main():
    context = zmq.Context()
    
    # Synchronization socket to receive readiness signals
    sync_socket = get_zmq_socket(context, zmq.REP, "ipc://sync.ipc")
    
    # Publisher socket
    socket = get_zmq_socket(context, zmq.PUB, "ipc://pubsub.ipc")
    
    # Spawn subscribers with integer identifiers 1 and 2
    sub1 = spawn_one_subscriber(1)
    sub2 = spawn_one_subscriber(2)
    
    # Wait for subscribers to signal readiness
    for _ in range(2):
        msg = sync_socket.recv()
        print("Received subscriber ready signal")
        sync_socket.send(b'')  # Send acknowledgment
    
    # Allow some time for subscribers to process the sync messages
    time.sleep(1)
    
    # Send messages to subscribers
    for i in range(5):
        msg1 = {'message': f"Message {i} to subscriber 1", 'data': b'\x00\x01\x02'}
        msg2 = {'message': f"Message {i} to subscriber 2", 'data': b'\x03\x04\x05'}
        topic1 = (1).to_bytes(1, byteorder='big')
        topic2 = (2).to_bytes(1, byteorder='big')
        print(f"Publishing to topic {1}: {msg1}")
        socket.send_multipart([topic1], zmq.SNDMORE)
        socket.send_pyobj(msg1)
        print(f"Publishing to topic {2}: {msg2}")
        socket.send_multipart([topic2], zmq.SNDMORE)
        socket.send_pyobj(msg2)
        time.sleep(0.5)
    
    # Send stop commands to subscribers
    print("Sending stop commands")
    stop_msg = {'command': 'STOP'}
    topic1 = (1).to_bytes(1, byteorder='big')
    topic2 = (2).to_bytes(1, byteorder='big')
    socket.send_multipart([topic1], zmq.SNDMORE)
    socket.send_pyobj(stop_msg)
    socket.send_multipart([topic2], zmq.SNDMORE)
    socket.send_pyobj(stop_msg)
    
    # Wait for subscribers to terminate
    sub1.join()
    sub2.join()
    
    socket.close()
    sync_socket.close()
    context.term()

if __name__ == "__main__":
    main()
