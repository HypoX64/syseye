import os
import time
# -*- coding:utf-8 -*- 

'''
-----------------------------Cpu-----------------------------
'''
# Cpu use
last_worktime=0
last_idletime=0
def get_cpu_use():
    global last_worktime, last_idletime
    f=open("/proc/stat","r")
    line=""
    while not "cpu " in line:
        line=f.readline()
    f.close()
    spl=line.split(" ")
    worktime=int(spl[2])+int(spl[3])+int(spl[4])
    idletime=int(spl[5])
    dworktime=(worktime-last_worktime)
    didletime=(idletime-last_idletime)
    rate=float(dworktime)/(didletime+dworktime)
    cpu_t = rate*100
    last_worktime=worktime
    last_idletime=idletime
    if(last_worktime==0): 
        return 0
    return round(cpu_t,2)

# Cpu freq
def get_cpu_freq():
    freq_str = os.popen('cat /proc/cpuinfo | grep MHz | uniq').read()
    freqs =[]
    while freq_str.find('\n') != -1:
        freqs.append(float(freq_str[freq_str.find(':')+2:freq_str.find('\n')]))
        freq_str=freq_str[freq_str.find('\n')+1:]
    return max(freqs)

# Cpu temp
def get_cpu_temp():
    if os.path.isfile('/sys/class/thermal/thermal_zone0/temp'):
        temp_str = os.popen('cat /sys/class/thermal/thermal_zone0/temp').read()
    elif os.path.isfile('/sys/class/hwmon/hwmon0/device/hwmon/hwmon0/temp1_input'):
        temp_str = os.popen('cat /sys/class/hwmon/hwmon0/device/hwmon/hwmon0/temp1_input').read()
    elif os.path.isfile('/sys/class/hwmon/hwmon0/device/hwmon0/temp1_input'):
        temp_str = os.popen('cat /sys/class/hwmon/hwmon0/device/hwmon0/temp1_input').read()
    else:
        return -1
    return (float(temp_str)/1000)

'''
-----------------------------Men-----------------------------
'''
def get_mem_use():
    mem_str = os.popen('free').read()
    if '内存'in mem_str :
        mem_str = mem_str[mem_str.find('内存：')+3:mem_str.find('交换')]
    else:
        mem_str = mem_str[mem_str.find('Mem:')+4:mem_str.find('Swap')]
    mem_str = mem_str.split()
    total = int(float(mem_str[0])/1024)
    used = int(float(mem_str[1])/1024)
    percent = int(used/total*100)
    return total,used,percent

def get_swap_use():
    mem_str = os.popen('free').read()
    if '内存'in mem_str :
        mem_str = mem_str[mem_str.find('交换：')+3:]
    else:
        mem_str = mem_str[mem_str.find('Swap:')+5:]
    mem_str = mem_str.split()
    total = int(float(mem_str[0])/1024)
    used = int(float(mem_str[1])/1024)
    percent = int(used/total*100)
    return total,used,percent

'''
-----------------------------Gpu-----------------------------
'''
gpus_str = os.popen('nvidia-smi -L').read()
gpus =[]
if 'communicate with the NVIDIA driver' not in gpus_str:
    while gpus_str.find('\n') != -1:
        gpus.append(gpus_str[gpus_str.find(':')+2:gpus_str.find('(')-1])
        gpus_str=gpus_str[gpus_str.find('\n')+1:]

def get_gpu_use():
    
    gpu_infos = []
    out_string = os.popen('nvidia-smi').read()
    # gpu infos
    for i in range(len(gpus)):
        infos_str = out_string
        infos_str = infos_str[infos_str.find(str(i)+'  '+gpus[i]):]
        infos_str = infos_str[infos_str.find('\n')+1:]
        infos_str = infos_str[:infos_str.find('\n')+1]
        infos_str = infos_str.split()
        #['|', '50%', '42C', 'P0', '19W', '/', '75W', '|', '929MiB', '/', '5050MiB', '|', '14%', 'Default', '|']
        if infos_str[1].replace('%','') == 'N/A':
            fan = -1
        else:
            fan = int(infos_str[1].replace('%',''))        # %
        temp = int(infos_str[2].replace('C',''))       # C
        if infos_str[4] == 'N/A':
            power_used = -1
            power_max = -1
        else:
            power_used = int(infos_str[4].replace('W','')) # W
            power_max = int(infos_str[6].replace('W',''))  # W
        mem_used = int(infos_str[8].replace('MiB','')) # MB
        mem_max = int(infos_str[10].replace('MiB',''))  # MB
        util_used = int(infos_str[12].replace('%',''))  # %
        gpu_infos.append([fan,temp,power_used,power_max,mem_used,mem_max,util_used])
    # cuda infos
    infos_str = out_string
    infos_str = infos_str.split('\n')
    for line in infos_str:
        if 'NVIDIA-SMI' in line:
           cuda_infos = '\033[1;37m'+line.replace('|','')[1:]+'\033[0m'
    
    # task_infos
    task_infos = '\033[1;37mGPU       PID   Type   Process name                        GPU Memory Usage\033[0m\n'
    infos_str = out_string
    infos_str = infos_str[infos_str.find('Processes'):]
    infos_str = infos_str[infos_str.find('======'):]
    infos_str = infos_str[infos_str.find('\n')+1:]
    infos_str = infos_str[:infos_str.find('+-------------------')-1]
    infos_str = infos_str.replace('|','')
    infos_str = infos_str.split('\n')
    print_flag = False
    for line in infos_str:
        line_split = line.split()
        if 'No running' not in line and float(line_split[4].replace('MiB',''))>500:
            task_infos = task_infos+line[2:]+'\n'
            print_flag = True
    if not print_flag:
        task_infos=''

    return gpu_infos,cuda_infos,task_infos


'''
-----------------------------Network-----------------------------
'''
def get_task_info():
    stream = os.popen('top -n 1')._stream
    out_string = stream.buffer.read().decode('utf-8',"ignore")
    out_string = out_string.split('\n')

    infos = ''
    if len(out_string) > 10:
        infos = '\033[1;37m  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND     \033[0m\n'
        for i in range(3):
            infos += (out_string[7+i]+'\n')

    return infos


'''
-----------------------------Network-----------------------------
'''
net_infos_history = [0,0]
def get_net_use(network_adapter = 'all'):
    net_str = os.popen('cat /proc/net/dev').read()
    infos = []
    net_infos = [0,0,0,0]
    while net_str.find('\n') != -1:
        if net_str[:50].find(':') != -1:
            infos.append((net_str[net_str.find(':')+2:net_str.find('\n')-1]).split())
        net_str=net_str[net_str.find('\n')+1:]
    net_rxs=[];net_txs=[]
    for info in infos:
        net_rxs.append(int(info[0]))
        net_txs.append(int(info[8]))
    if network_adapter == 'all':
        net_infos[0] = (sum(net_rxs)/1024)
        net_infos[1] = (sum(net_txs)/1024)
    elif network_adapter == 'auto':
        net_infos[0] = (max(net_rxs)/1024)
        net_infos[1] = (max(net_txs)/1024)
    if net_infos_history[0] == 0:
        net_infos[2] = 0
        net_infos[3] = 0
    else:
        net_infos[2] = net_infos[0]-net_infos_history[0]
        net_infos[3] = net_infos[1]-net_infos_history[1]
    net_infos_history[0] = net_infos[0]
    net_infos_history[1] = net_infos[1]
    return net_infos

'''
-----------------------------Disk-----------------------------
'''
def get_disk_use():
    disk_str = os.popen('df -h').read()
    disk_str = disk_str.split('\n')
    disk_infos = []
    # print(disk_str)
    allows = ['/home','/media']  # and '/'
    for line in disk_str:
        for allow in allows:
            if allow in line[38:] or len(line)==(line[25:].find('/')+25+1):   
                info = line.split()
                info[4] = auto_color(info[4], int(info[4].replace('%','')))
                disk_infos.append(info)
                break
    return disk_infos

'''
-----------------------------other tools-----------------------------
'''
def get_bar(percent,num = 25):
    bar = '['
    for i in range(num):
        if i < round(percent/(100/num)):
            bar += '#'
        else:
            bar += '-'
    bar += ']'
    bar = '{0:>5.1f}% '.format(percent)+bar
    bar = auto_color(bar, percent)
    return bar

def fill_str(string,num):
    if len(string)>num:
        return string[:num-3]+'...'
    for i in range(num-len(string)):
        string +=' '
    return string

def auto_color(string,percent):
    if 0<=percent<=70:
        string = change_color(string, 'green')
    elif 70<percent<=90:
        string = change_color(string, 'yellow')
    else:
        string = change_color(string, 'red')
    return string

def change_color(string,color):
    if color =='red':
        string = '\033[1;31m'+string+'\033[0m'
    elif color == 'green':
        string = '\033[1;32m'+string+'\033[0m'
    elif color == 'yellow':
        string = '\033[1;33m'+string+'\033[0m'
    if color =='white':
        string = '\033[1;37m'+string+'\033[0m'
    return string

'''
-----------------------------main-----------------------------
'''
def main():
    while(1):
        #cpu
        cpu_used = get_cpu_use()
        cpu_freq = get_cpu_freq()
        cpu_temp = get_cpu_temp()
        cpu_used_bar = get_bar(cpu_used,num=65)
        task_infos = get_task_info()

        #memory
        mem_total,mem_used,mem_percent = get_mem_use()
        mem_used_bar = get_bar(mem_percent)
        swap_total,swap_used,swap_percent = get_swap_use()
        swap_used_bar = get_bar(swap_percent)

        #gpu
        util_used_bars=[];gpu_mem_bars=[]
        gpu_infoss,cuda_infos,gpu_task_infos = get_gpu_use()
        for i in range(len(gpus)):
            gpu_infos = gpu_infoss[i]
            util_used_bars.append(get_bar(gpu_infoss[i][6]))
            gpu_mem_bars.append(get_bar(100*gpu_infoss[i][4]/gpu_infoss[i][5]))

        #net
        net_infos = get_net_use()

        #disk
        disk_infos = get_disk_use()

        #-----------------------------print-----------------------------
        os.system('clear')
        #cpu memory
        print(('\033[1;37mCpu-T: {0:.1f}C | Freq: {1:.1f}MHz | Mem: {2:d}MB/{3:d}MB | Swap: {4:d}MB/{5:d}MB\033[0m').format(
            cpu_temp,cpu_freq,mem_used,mem_total,swap_used,swap_total))
        print('Cpu: '+cpu_used_bar)
        print('Mem: '+mem_used_bar+' Swap:'+swap_used_bar)
        
        # Task
        print(task_infos)
        
        #gpu
        print(cuda_infos)
        for i in range(len(gpus)):
            print(('\033[1;37mGpu'+'{0:d}'+': '+gpus[i].replace('GeForce','')+'  Temp: {1:.1f}C | Power: {2:>3d}w/{3:d}w | Mem: {4:>5d}MB/{5:d}MB | Fan: {6:d}%\033[0m').format(
                i,gpu_infoss[i][1],gpu_infoss[i][2],gpu_infoss[i][3],
                gpu_infoss[i][4],gpu_infoss[i][5],gpu_infoss[i][0]))
            print('Util:'+util_used_bars[i]+'  Mem:'+gpu_mem_bars[i])
        print(gpu_task_infos)
        
        #net
        print(('\033[1;37mNetwork    ↑ all:{0:.1f}GB ↓ all:{1:.1f}GB     ↑ :{2:.1f}Kb/s ↓ :{3:.1f}Kb/s\033[0m').format(
            net_infos[1]/1024/1024,net_infos[0]/1024/1024,net_infos[3],net_infos[2]))

        #disk
        print('\n\033[1;37mFilesystem           Mounted on           Used/Total           Used%\033[0m')
        for disk_info in disk_infos:
            print(fill_str(disk_info[0], 20),fill_str(disk_info[5], 20),
                fill_str(disk_info[2]+'/'+disk_info[1], 20),fill_str(disk_info[4], 15))

        time.sleep(1)

if __name__ == '__main__':
    main()
