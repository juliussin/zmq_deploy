import argparse
import zmq


def main(port_in, port_out):
    """ Main method """

    context = zmq.Context()

    # Socket facing clients
    frontend = context.socket(zmq.ROUTER)
    frontend.bind("tcp://*:{}".format(port_in))

    # Socket facing services
    backend = context.socket(zmq.DEALER)
    backend.bind("tcp://*:{}".format(port_out))

    print(f"Proxy initialized: ROUTER bound to port {port_in}, DEALER bound to port {port_out}")
    zmq.proxy(frontend, backend)

    # We never get here…
    frontend.close()
    backend.close()
    context.term()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="ZMQ Proxy Router-Dealer example with configurable ports."
    )
    parser.add_argument(
        "--port-in",
        type=int,
        required=True,
        help="Port for incoming connections (ROUTER socket).",
    )
    parser.add_argument(
        "--port-out",
        type=int,
        required=True,
        help="Port for outgoing connections (DEALER socket).",
    )

    args = parser.parse_args()

    # Run the main method with parsed arguments
    main(args.port_in, args.port_out)
