import math
import sys

n = int(sys.argv[1])
dims = int(math.log2(n))

print("<?xml version='1.0'?>")
print('<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">')
print('<platform version="4.1">')
print('  <zone id="world" routing="Floyd">')

for i in range(n):
    print(f'    <host id="node-{i}.simgrid.org" speed="100Gf"/>')

for i in range(n):
    for bit in range(dims):
        j = i ^ (1 << bit)
        if i < j and j < n:
            print(f'    <link id="link-{i}-{j}" bandwidth="12.5GBps" latency="1us"/>')

for i in range(n):
    for bit in range(dims):
        j = i ^ (1 << bit)
        if i < j and j < n:
            print(f'    <route src="node-{i}.simgrid.org" dst="node-{j}.simgrid.org">')
            print(f'      <link_ctn id="link-{i}-{j}"/>')
            print("    </route>")

print("  </zone>")
print("</platform>")
