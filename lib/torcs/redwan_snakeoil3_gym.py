import os
import time
import getopt
import socket
import sys
from .snakeoil3_gym import ServerState,DriverAction



_practice_path = '/home/averma/torcs/torcs-1.3.7/src/raceman/practice.xml'
# Initialize help messages
ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

class Client(object):
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False, track='practice.xml'):

        #################################### modified by redwan
        self.__gui = vision
        self.__timeout = 10000
        self.__data_size = 2 ** 17
        self.__server = Server(vision,track)
        self.__socket = self.__create_socket()




        # If you don't like the option defaults,  change them here.
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 100000  # 50steps/second

        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d

        self.__quickrace_xml_path = os.path.expanduser('~') + '/.torcs/config/raceman/{}'.format(track)
        self.S= ServerState()
        self.R= DriverAction()


        self.__connect_to_server()
        # self.parse_the_command_line()


    def __init_server(self):
        os.system('pkill torcs')
        time.sleep(0.001)
        if self.__gui:
            # if self.__cmd_exists('optirun'):
            #     os.system('optirun torcs -nofuel -nolaptime -s -t {} >/dev/null &'.format(self.__timeout))
            # else:
            os.system('torcs -nofuel -nolaptime -s -t {} >/dev/null &'.format(self.__timeout))
            time.sleep(2)
            os.system('sh utilities/autostart.sh')
        else:
            os.system('torcs -nofuel -nolaptime -t 50000 -r '.format(
                self.__timeout) + self.__quickrace_xml_path + ' >/dev/null &')
        # print('Server created!')
        time.sleep(0.001)

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                                         ['host=', 'port=', 'id=', 'steps=',
                                          'episodes=', 'track=', 'stage=',
                                          'debug', 'help', 'version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug = True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host = opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid = opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname = opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage = int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port = int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes = int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps = int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)



    def __connect_to_server(self):
        tries = 3
        while True:
            sensor_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, sensor_angles)

            try:
                self.__socket.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()

            try:
                sockdata, address = self.__socket.recvfrom(self.__data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                # print("Waiting for __server on __port " + str(self.__port))
                tries -= 1
                if tries == 0:
                    # print("Server didn't answer, sending restart signal")
                    self.__server.restart()

            identify = '***identified***'
            if identify in sockdata:
                # print("Client connected on __port " + str(self.__port))
                break

    def __send_message(self, message):
        try:
            self.__socket.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print(u"Error sending to __server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)

    def get_servers_input(self):
        sockdata = str()
        '''Server's input is stored in a ServerState object'''
        if not self.__socket: return
        sockdata = str()

        while True:
            try:
                # Receive server data
                sockdata, addr = self.__socket.recvfrom(self.__data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print('.', end=' ')
                # print "Waiting for data on %d.............." % self.port
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. " +
                        "You were in %d place.") %
                       (self.port, self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                # What do I do here?
                print("Server has restarted the race on %d." % self.port)
                # I haven't actually caught the server doing this.
                self.shutdown()
                return
            elif not sockdata:  # Empty?
                continue  # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")  # Clear for steady output.
                    print(self.S)
                break  # Can now return from this function.
            if sockdata:
                return self.__parse_server_string(sockdata)

        # while True:
        #     try:
        #         sockdata, address = self.__socket.recvfrom(self.__data_size)
        #         sockdata = sockdata.decode('utf-8')
        #     except socket.error:
        #         print('', end='')
        #     if sockdata:
        #         return self.__parse_server_string(sockdata)
    def __parse_server_string(self, server_string):
            track_data = {}
            server_string = server_string.strip()[:-1]
            server_string_list = server_string.strip().lstrip('(').rstrip(')').split(')(')
            for i in server_string_list:
                w = i.split(' ')
                track_data[w[0]] = self.__destringify(w[1:])
            return track_data

    def shutdown(self):
        if not self.__socket: return
        print(("Race terminated or %d steps elapsed. Shutting down %d."
               % (self.maxSteps,self.port)))
        self.__socket.close()
        self.R.d['meta'] = 1
        self.__socket = None

    def respond_to_server(self):
        if not self.__socket: return
        try:
            message = repr(self.R)
            self.__socket.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def __destringify(self, string):
        if not string:
            return string
        if type(string) is str:
            try:
                return float(string)
            except ValueError:
                print("Could not find a value in %s" % string)
                return string
        elif type(string) is list:
            if len(string) < 2:
                return self.__destringify(string[0])
            else:
                return [self.__destringify(i) for i in string]

    @property
    def restart(self):
        # print('Restarting __server...')
        self.__socket = self.__create_socket()
        return self.__connect_to_server()


    @staticmethod
    def __create_socket():
        try:
            so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create __socket...')
            sys.exit(-1)
        so.settimeout(1)
        return so


class Server:

    def __init__(self, gui, track, timeout=10000):
        self.__gui = gui
        self.__quickrace_xml_path = os.path.expanduser('~') + '/.torcs/config/raceman/{}'.format(track)
        # self.__create_race_xml(track, track_type)
        self.__timeout = timeout
        self.__init_server()

    # @staticmethod
    # def __cmd_exists(cmd):
    #     return subprocess.call("type " + cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

    def __init_server(self):
        os.system('pkill torcs')
        time.sleep(0.001)
        if self.__gui:
            # if self.__cmd_exists('optirun'):
            #     os.system('optirun torcs -nofuel -nolaptime -s -t {} >/dev/null &'.format(self.__timeout))
            # else:
            os.system('torcs -nofuel -nolaptime -s -t {} >/dev/null &'.format(self.__timeout))
            time.sleep(2)
            os.system('sh utilities/autostart.sh')
        else:
            os.system('torcs -nofuel -nolaptime -t 50000 -r '.format(self.__timeout) + self.__quickrace_xml_path + ' >/dev/null &')
        # print('Server created!')
        time.sleep(0.001)

    def restart(self):
        # print('Restarting __server...')
        self.__init_server()

