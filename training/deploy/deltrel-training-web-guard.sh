#!/bin/sh
# Preserve the training host's private web ports when the provider's shared
# workspace rules permit HTTP/HTTPS for the separate playing instance.
set -eu

for filter_binary in /usr/sbin/iptables /usr/sbin/ip6tables; do
    if ! "$filter_binary" -w 10 -C INPUT ! -i lo -p tcp \
        -m multiport --dports 80,443 \
        -m comment --comment deltrel-training-private-web -j DROP 2>/dev/null; then
        "$filter_binary" -w 10 -I INPUT 1 ! -i lo -p tcp \
            -m multiport --dports 80,443 \
            -m comment --comment deltrel-training-private-web -j DROP
    fi
done
