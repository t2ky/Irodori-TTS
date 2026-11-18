#!/usr/bin/env bash
# stop whatever listens on the port (default 8099); pkill -f would also match the caller's shell
port=${1:-8099}
pid=$(ss -ltnpH "sport = :$port" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
[ -n "$pid" ] && kill "$pid" && while kill -0 "$pid" 2>/dev/null; do sleep 0.5; done
echo "stopped ${pid:-nothing} on $port"
