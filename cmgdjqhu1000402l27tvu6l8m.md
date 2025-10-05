---
title: "basic namespace"
datePublished: Sun Oct 05 2025 10:15:36 GMT+0000 (Coordinated Universal Time)
cuid: cmgdjqhu1000402l27tvu6l8m
slug: basic-namespace
tags: basicnamespace

---

The **basic\_space** example demonstrates how the Linux kernel provides each process with a completely independent network space (*network namespace*), thereby experimentally illustrating the principles of **“spatial isolation”** and **“linking”**.

Because the original examples provided by *tinet* often cause issues such as duplicate container names or IP conflicts,  
it is recommended to use the example located in the

`examples/basic_namespace` directory of the following GitHub repository instead.

[https://github.com/eumgil0812/network](https://github.com/eumgil0812/network)

```bash

tinet up -c spec.blue.yaml > up_blue.sh
chmod +x up_blue.sh
sudo ./up_blue.sh

tinet up -c spec.green.yaml > up_green.sh
chmod +x up_green.sh
sudo ./up_green.sh
```

```bash
docker exec -it blue_R1 ip addr add 20.0.0.1/24 dev net0
docker exec -it blue_R2 ip addr add 20.0.0.2/24 dev net0
docker exec -it green_R1 ip addr add 10.0.0.1/24 dev net0
docker exec -it green_R2 ip addr add 10.0.0.2/24 dev net0
```

The tinet developers have drawn a cute diagram to show it.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759653322271/3a9c74a2-c835-4cb6-a23d-703ade57677e.png align="center")

### 🔍 Key Points

1. **Two Independent Network Worlds**
    
    * The blue box (`namespace: blue`) and the green box (`namespace: green`) represent  
        **completely isolated network environments**.
        
    * They exist on the same system but cannot see or interact with each other —  
        they’re separate *virtual network worlds*.
        
2. **Same Structure, Different IP Ranges**
    
    * Both namespaces have the same topology (`R1 ↔ R2`),  
        but use different IP networks:
        
        * `blue`: `20.0.0.0/24`
            
        * `green`: `10.0.0.0/24`
            
    * This allows identical interface names (like `net0`) or similar IPs to be reused  
        without any conflict between namespaces.
        
3. **Purpose of the Example**
    
    * It experimentally demonstrates how the Linux kernel can **isolate network stacks per process**.  
        For example, `ping 20.0.0.2` only works *inside* the blue namespace —  
        the green namespace has no idea that it exists.
        

---

### 🧠 What is a *namespace*?

A **namespace** is a Linux kernel feature that lets you create multiple isolated virtual environments within a single system.  
You can isolate:

* processes
    
* file systems
    
* network interfaces
    
* and more
    

The **network namespace** specifically isolates **network resources** such as interfaces, IP addresses, and routing tables.  
Each process can therefore behave as if it has its *own* network stack.

## ✅ **Test 1. Verify Communication Within the Same Namespace Group**

🎯 **Objective**

Communication between **blue\_R1 ↔ blue\_R2** must succeed.  
Communication between **green\_R1 ↔ green\_R2** must also succeed.

```bash
docker exec -it blue_R1 ping -c 2 20.0.0.2
docker exec -it green_R1 ping -c 2 10.0.0.2
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759653592434/25bc810b-d67e-47d8-af69-844241e83d8b.png align="center")

✅ **Success**  
→ This means that the **veth pairs** inside each namespace are properly connected,  
→ and both **IP assignment** and **link status** are functioning correctly.

## 🚫 **Test 2. Attempt Cross-Namespace Communication (Isolation Check)**

🎯 **Objective**

Communication between **different namespaces** must **not** succeed under any circumstances.

```bash
docker exec -it blue_R1 ping -c 2 10.0.0.1
docker exec -it green_R1 ping -c 2 20.0.0.1
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759653668310/fa87f77d-25a1-4289-920b-57ccb5d3b06b.png align="center")

❌ **Failure Expected (Normal Behavior)**  
This happens because the **Linux kernel completely isolates the network stack** for each namespace.  
As a result, namespaces cannot communicate with each other unless explicitly connected via a virtual link or bridge.

## 🧠 **Test 3. IP Conflict Scenario**

🎯 **Goal**  
Deliberately assign **the same IP subnet (e.g., 10.0.0.1/24)** to both namespaces and observe what happens.

```bash
docker exec -it blue_R1 ip addr flush dev net0
docker exec -it blue_R2 ip addr flush dev net0
docker exec -it blue_R1 ip addr add 10.0.0.1/24 dev net0
docker exec -it blue_R2 ip addr add 10.0.0.2/24 dev net0
```

```bash
docker exec -it blue_R1 ping -c 2 10.0.0.2
docker exec -it green_R1 ping -c 2 10.0.0.2
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759653804988/474fa32f-42c4-4f6c-9684-5b0e0f736a3b.png align="center")

Surprise!

✅ **Both succeeded**

Because —  
Even within the same Linux kernel, **IP addresses do not conflict across different namespaces.**  
In other words, you can have **two independent interfaces both using** `10.0.0.1` at the same time without any issue.

👉 This experiment is **definitive proof** that a **network namespace is a completely isolated, virtual networking environment.**

# 🚀 **Deep Dive: Crossing the Boundary (Advanced)**

Now, you can create a **bridge** or a **router** to connect the isolated `blue` and `green` network namespaces.  
This enables a **full cross-namespace routing experiment**, where packets can traverse from one virtual network to another —  
just like real-world inter-network communication through a gateway.

In other words:

> You’re about to move from *isolation testing* 🧱 → to *inter-network connectivity* 🌐.

## 1.bridge 방법 사용하기

### 1️⃣ **Re-register Network Namespaces**

Retrieve the **PID** of each Docker container and re-link their **network namespaces** under `/var/run/netns`.

This allows you to access each container’s network stack using `ip netns exec`, just like a standard Linux network namespace.

```bash
PID=$(docker inspect blue_R1 --format '{{.State.Pid}}')
ln -sf /proc/$PID/ns/net /var/run/netns/blue_R1

PID=$(docker inspect green_R1 --format '{{.State.Pid}}')
ln -sf /proc/$PID/ns/net /var/run/netns/green_R1
```

```bash
ip netns list
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759655374717/b4949073-5899-4c70-af65-2c33f8ce6a00.png align="center")

### 2️⃣Veth check

```bash
ip link add veth-blue_R1 type veth peer name veth-green_R1
ip link set veth-blue_R1 netns blue_R1
ip link set veth-green_R1 netns green_R1
```

```bash
ip netns exec blue_R1 ip link
ip netns exec green_R1 ip link.
```

### 3️⃣**Attach the links and assign IP addresses**

```bash
ip netns exec blue_R1 ip link set veth-blue_R1 up
ip netns exec green_R1 ip link set veth-green_R1 up
ip netns exec blue_R1 ip addr add 192.168.0.1/24 dev veth-blue_R1
ip netns exec green_R1 ip addr add 192.168.0.2/24 dev veth-green_R1
```

### 4️⃣ Connecting Test

```bash
ip netns exec blue_R1 ping -c 2 192.168.0.2
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759655752800/55b07d10-ba26-4433-b1e1-eace89476686.png align="center")

✅ ping success!

## 2.Router

### 🌐 Goal: ASCII Topology

```bash
┌────────────────────────────┐                ┌────────────────────────────┐
│        blue_R1             │                │         green_R1           │
│  (10.0.0.1/24)             │                │   (20.0.0.1/24)           │
│                            │                │                            │
│        [veth-b]            │                │          [veth-g]          │
└────────────┬───────────────┘                └────────────┬───────────────┘
             │                                               │
             │                                               │
             ▼                                               ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                        R_ROUTER                             │
       │-------------------------------------------------------------│
       │  [veth-br] 10.0.0.254/24      [veth-gr] 20.0.0.254/24       │
       │       (blue network)             (green network)             │
       │        ▲   ↕    ip_forward=1    ▲   ↕                       │
       │       └────────── Routing Table  ───────────┘                 │
       └─────────────────────────────────────────────────────────────┘
```

## 🧠 **Key Concepts Summary**

| Component | Description |
| --- | --- |
| **blue\_R1** | A host in the **10.0.0.0/24** network (e.g., PC A) |
| **green\_R1** | A host in the **20.0.0.0/24** network (e.g., PC B) |
| **R\_ROUTER** | A namespace acting as a **router** connecting the two networks |
| **veth-b / veth-br** | Virtual cable connecting **blue\_R1 ↔ R\_ROUTER** |
| **veth-g / veth-gr** | Virtual cable connecting **green\_R1 ↔ R\_ROUTER** |
| **IP Forwarding** | Allows the kernel to forward packets between interfaces |
| **Routing Table** | A map that determines which interface to use for a given destination network |

Here, **blue\_R2** and **green\_R1** each act as a **router** (container or namespace) that intermediates between the two networks.

### 🧱 1️⃣ router + green\_R1

```bash
# Router Container
docker run -td --net none --name router --rm --privileged --hostname router -v /tmp/tinet:/tinet slankdev/ubuntu:18.04 > /dev/null
PID=$(docker inspect router --format '{{.State.Pid}}')
ln -sf /proc/$PID/ns/net /var/run/netns/router

# Green Container
docker run -td --net none --name green_R1 --rm --privileged --hostname green_R1 -v /tmp/tinet:/tinet slankdev/ubuntu:18.04 > /dev/null
PID=$(docker inspect green_R1 --format '{{.State.Pid}}')
ln -sf /proc/$PID/ns/net /var/run/netns/green_R1
.
```

### ⚙️ 3️⃣ name space

First, let's create three namespaces:

```bash
ip netns add blue_R1
ip netns add R_ROUTER
ip netns add green_R1
```

### 🪄 4️⃣ **Set up the links**

```bash
# blue_R1 ↔ R_ROUTER Connect
ip link add veth-b type veth peer name veth-br
ip link set veth-b netns blue_R1
ip link set veth-br netns R_ROUTER

# green_R1 ↔ R_ROUTER Connect
ip link add veth-g type veth peer name veth-gr
ip link set veth-g netns green_R1
ip link set veth-gr netns R_ROUTER
```

### 🌐 5️⃣ IP Address

```bash
# Blue namespace
ip netns exec blue_R1 ip addr add 10.0.0.1/24 dev veth-b
ip netns exec blue_R1 ip link set veth-b up
ip netns exec blue_R1 ip link set lo up

# Green namespace
ip netns exec green_R1 ip addr add 20.0.0.1/24 dev veth-g
ip netns exec green_R1 ip link set veth-g up
ip netns exec green_R1 ip link set lo up

# Router namespace
ip netns exec R_ROUTER ip addr add 10.0.0.254/24 dev veth-br
ip netns exec R_ROUTER ip addr add 20.0.0.254/24 dev veth-gr
ip netns exec R_ROUTER ip link set veth-br up
ip netns exec R_ROUTER ip link set veth-gr up
ip netns exec R_ROUTER ip link set lo up
```

### 🚀 6️⃣ Routing Setting

```bash
# IP forwarding 
ip netns exec R_ROUTER sysctl -w net.ipv4.ip_forward=1

# Add rout from blue_R1 to green_R1
ip netns exec blue_R1 ip route add 20.0.0.0/24 via 10.0.0.254

# Add rout from green_R1 to blue_R1
ip netns exec green_R1 ip route add 10.0.0.0/24 via 20.0.0.254
```

### ✅ 7️⃣ Connecting Test

```bash
ip netns exec blue_R1 ping -c 3 20.0.0.1
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759658251960/35101125-1fab-4c95-99bf-5e065c4f5220.png align="center")