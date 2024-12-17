script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

portIn=3168     # client (send img -> get pred)
portOut=3169    # server (get img -> send pred)

python3 broker.py --port-in $portIn --port-out $portOut & 
python3 run_server_zmq.py --port-out $portOut
