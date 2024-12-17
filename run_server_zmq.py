import argparse
import time
import cv2
import numpy as np
import zmq
from mmdeploy_runtime import Classifier

def parse_arguments():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description="ZMQ Classifier Server for TensorRT")
    parser.add_argument("--port-out", type=int, required=True, help="ZMQ REP socket port to connect to")
    parser.add_argument("--model-path", type=str, default="/home/src/mmdeploy_model", help="Path to the model")
    parser.add_argument("--device-name", type=str, default="cuda", help="Device to run inference on (e.g., cuda or cpu)")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for CUDA")
    return parser.parse_args()

def initialize_model(model_path, device_name, device_id):
    """ Initialize the model with error handling """
    try:
        print(f"Loading model from: {model_path} on {device_name}:{device_id}")
        model = Classifier(model_path=model_path, device_name=device_name, device_id=device_id)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def setup_zmq_server(port_out):
    """ Set up the ZMQ REP socket server """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(f"tcp://127.0.0.1:{port_out}")
    print(f"ZMQ Server connected on tcp://127.0.0.1:{port_out}")
    return socket

def process_frame(frame, frame_shape):
    """ Convert received frame into numpy array """
    try:
        h_img, w_img, dim_img = map(int, frame_shape.decode().split("_"))
        if h_img * w_img == 0:
            return None
        return np.frombuffer(frame, dtype=np.uint8).reshape((h_img, w_img, dim_img))
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def main():
    """ Main server function """
    args = parse_arguments()

    # Initialize model and server
    model = initialize_model(args.model_path, args.device_name, args.device_id)
    socket = setup_zmq_server(args.port_out)

    print("Classifier Deployment Server Start.. !!")
    while True:
        try:
            # 1. Receive Frame
            msg_recv = socket.recv_multipart()
            frame, frame_shape = msg_recv

            # 2. Parse Frame
            parsed_frame = process_frame(frame, frame_shape)
            if parsed_frame is None:
                arr_results = np.array([[]])
            else:
                prev_time = time.time()
                print(f"Frame shape: {parsed_frame.shape}")

                # 3. Predict
                try:
                    result = model(parsed_frame)
                    arr_results = np.array([[label, score] for label, score in result]).astype(np.float64)
                except Exception as e:
                    print(f"Error during model inference: {e}")
                    arr_results = np.array([[]])

                fps = int(1 / (time.time() - prev_time))
                print(f"Deployment TensorRT FPS: {fps}")

            # 4. Send Results
            socket.send(arr_results)
        except KeyboardInterrupt:
            print("Server interrupted. Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
