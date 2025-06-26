from flask import Flask, render_template, request
import subprocess
import threading
import time
from werkzeug.utils import secure_filename
import os
import subprocess
import mmap
import signal
import serial


job_status = {"running": False, "result": "", "thread": None}

app = Flask(__name__)

port = '/dev/ttyUSB3'
#port = '/dev/ttyUSB2'
baudrate = '921600'
#baudrate = '115200'
exe_path = "/usr/bin/tsi/v0.1.1*/bin/"

DEFAULT_REPEAT_PENALTY = 1.5
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_LAST_N = 5
DEFAULT_CONTEXT_LENGTH = 12288
DEFAULT_TEMP = 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/llama-cli', methods=['GET'])
def llama_cli_serial_command():

    #./run_llama_cli.sh "my cat's name" "10" "tinyllama-vo-5m-para.gguf" "none"
    model = request.args.get('model')
    backend = request.args.get('backend')
    tokens = request.args.get('tokens')
    prompt = request.args.get('prompt')
    repeat_penalty = request.args.get('repeat-penalty', DEFAULT_REPEAT_PENALTY)
    batch_size = request.args.get('batch-size', DEFAULT_BATCH_SIZE)
    top_k = request.args.get('top-k', DEFAULT_TOP_K)
    top_p = request.args.get('top-p', DEFAULT_TOP_P)
    last_n = request.args.get('last-n', DEFAULT_LAST_N)
    context_length = request.args.get('context-length', DEFAULT_CONTEXT_LENGTH)
    temp = request.args.get('temp', DEFAULT_TEMP)

    # Define the model path (update with actual paths)
    model_paths = {
        "tiny-llama": "tinyllama-vo-5m-para.gguf",
        "Tiny-llama-F32": "Tiny-Llama-v0.3-FP32-1.1B-F32.gguf"
    }

    model_path = model_paths.get(model, "")
    if not model_path:
        return f"<h2>Error: Model path not found for '{model}'</h2>"

    # Build llama-cli command
    #command = [
    #    "./llama-cli",
    #    "-p", prompt,
    #    "-m", model_path,
    #    "--device", backend,
    #    "--temp", "0",
    #    "--n-predict", tokens,
    #    "--repeat-penalty", "1",
    #    "--top-k", "0",
    #    "--top-p", "1"
    #]
    # URL to Test this end point is as follows
    # http://10.50.30.167:5001/llama-cli?model=tiny-llama&backend=tSavorite&tokens=5&prompt=Hello+How+are+you
    script_path = "./run_llama_cli.sh"
    command = f"cd {exe_path}; {script_path} \"{prompt}\" {tokens} {model_path} {backend} {repeat_penalty} {batch_size} {top_k} {top_p} {last_n} {context_length} {temp}"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

UPLOAD_FOLDER = './' # Directory where recvFromHost is loaded 
destn_path='/tsi/proj/model-cache/gguf/' # Destination Directory in FPGA where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create the upload folder if it doesn't exist

@app.route('/upload-gguf', methods=['POST', 'GET'])
def upload_serial_command():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return "No file selected"

        # Save the file if it exists
        if file:
            filename = secure_filename(file.filename)
            process = subprocess.Popen(["./copy2fpga-x86.sh", filename], text=True)
            copy2fpgax86prints = "Starting copy2fpga-x86 and sending file..."
            print (copy2fpgax86prints)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            script_path = "./recvFromHost "
            command = f"cd {exe_path}; {script_path} {destn_path}{filename}"
            def scriptRecvFromHost():
                 try:
                     result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True,     check=True)
                     job_status["result"] = result.stdout
                     print(result.stdout)
                     recv_output = result.stdout
                 except subprocess.CalledProcessError as e:
                     job_status["result"] = f"Error: {e.stderr}"
                 finally:
                     job_status["running"] = False
            thread = threading.Thread(target=scriptRecvFromHost)
            job_status = {"running": True, "result": "", "thread": thread}
            thread.start()

            stdout, stderr = process.communicate()
        return render_template('uploadtofpga.html', apple = process, recvoutput=f"On FPGA Target, recvFromHost completed ; transf    ered file:{filename} received")
    return render_template('upload.html') # Display the upload form


#    command = f"upload file"
#    try:
#        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
#        return result.stdout, 200
#    except subprocess.CalledProcessError as e:
#        return f"Error executing script: {e.stderr}", 500

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return "No file selected"

        # Save the file if it exists
        if file:
            filename = secure_filename(file.filename)
            process = subprocess.Popen(["./copy2fpga-x86.sh", filename], text=True)
            copy2fpgax86prints = "Starting copy2fpga-x86 and sending file..."
            print (copy2fpgax86prints)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            script_path = "./recvFromHost "
            temporary_destination_path = request.form.get("destination_file_path") # I've tested this on fpga4 and it correctly gets the user-inputted file path
            command = f"cd {exe_path}; {script_path} {temporary_destination_path}{filename}"
            def scriptRecvFromHost():
                 try:
                     result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True,     check=True)
                     job_status["result"] = result.stdout
                     print(result.stdout)
                     recv_output = result.stdout
                 except subprocess.CalledProcessError as e:
                     job_status["result"] = f"Error: {e.stderr}"
                 finally:
                     job_status["running"] = False
            thread = threading.Thread(target=scriptRecvFromHost)
            job_status = {"running": True, "result": "", "thread": thread}
            thread.start()
 
            stdout, stderr = process.communicate()
        return render_template('uploadtofpga.html', apple = process, recvoutput=f"On FPGA Target, recvFromHost completed ; transf    ered file:{filename} received")
    return render_template('upload.html') # Display the upload form

@app.route('/restart-txe', methods=['GET'])
def restart_txe_serial_command():
    '''
    #THIS IS ASHISH'S OLD CODE
    command = f"telnet localhost 8000\r\nclose all\r\n"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        time.sleep(5)
        command = f"{exe_path}/../install/tsi-start\nyes\n"
        try:
            result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
            return result.stdout, 200
        except subprocess.CalledProcessError as e:
            return f"Error executing script: {e.stderr}", 500
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500
    #THIS IS ASHISH'S OLD CODE
    '''
    '''
    #THIS IS MY CODE BUT IT'S INCONSISTENT BECAUSE OF SERVER BUG!
    command = f"telnet localhost 8000\r\n"
    try:
        result = subprocess.run(['python3','serial_script.py',port,baudrate,command],capture_output=True,text=True,check=True)
        time.sleep(10)
        command = f"close all\r\n"
        try:
            result = subprocess.run(['python3','serial_script.py',port,baudrate,command],capture_output=True,text=True,check=True)
            time.sleep(5)
            command = f"{exe_path}/../install/tsi-start\n"
            try:
                result = subprocess.run(['python3','serial_script.py',port,baudrate,command],capture_output=True,text=True,check=True)
                time.sleep(15) #Changed to 15 just to be safe!
                #command = f"yes\n" Ashish said default was yes so just timeout is needed!
                try:
                    result = subprocess.run(['python3','serial_script.py',port,baudrate,command],capture_output=True,text=True,check=True)
                    return result.stdout, 200
                except subprocess.CalledProcessError as e:
                    return f"Error executing script: {e.stderr}", 500
            except subprocess.CalledProcessError as e:
                return f"Error executing script: {e.stderr}", 500
        except subprocess.CalledProcessError as e:
            return f"Error executing script: {e.stderr}", 500
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500
    #THIS IS MY CODE BUT IT'S INCONSISTENT BECAUSE OF SERVER BUG!
    '''
    #THIS WILL BE THE CODE FOR FULL REBOOT!
    #BY DEFAULT WE ALWAYS START IN SERIAL/PICOCOM
    #1. WE NEED TO BACK TO THE SHELL AND DO cd /tsi/fpga_card/fpga4/SKYLP_G0221/rev4; sudo make all; make juart
    #2. IN SERIAL/PICOCOM, WE NEED TO DO boot
    #UNLESS CALLED WE ARE ARE ALWAYS RUNNING IN CURRENT SHELL CONTEXT, WE ONLY WRITE TO SERIAL USING SUBPROCESS.RUN USING PORT AND BAUDRATE
    print("alskdjflskjfdlsadkdjf")
    command = f"cd /tsi/fpga_card/fpga4/SKYLP_G0221/rev4; sudo make all; make juart"
    
    '''
    process = subprocess.Popen([command],shell=True,preexec_fn=os.setsid)#This isn't finishing because its stuck on the juart terminal!
    try:
        process.wait(timeout=100)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)  # Optionally wait a bit more for clean exit
    '''
    process = subprocess.Popen([command],shell=True,preexec_fn=os.setsid,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        for line in process.stdout:
            #print(line, end="")  # Or process the line however you want
            if "release chip from reset called" in line:
                time.sleep(2)
                #print(f"Keyword '{keyword}' found. Terminating process...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                break
    except KeyboardInterrupt:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)






    #EVERYTHING ABOVE THIS WORKS! THE JUART TERMINAL AUTO EXITS AND IT REACHES THE PRINT STATEMENT BELOW!!!!!!!!!


    #time.sleep(100) #Give Popen time to set up everything and 
    
    print("reached here")
    
    #render_template("processing.html")
    
    #time.sleep(150)
    
    # These 2 things work; the big problem is both parts work separately but not together!
    ser = serial.Serial(port, baudrate)
    
    ser.write(b'boot\n')

    #time.sleep(50)

    #For this part to work you can't be in serial because multiple people can write, but only 1 person can read at a time!
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        #print(f"Received: {line}")
        if line:
            #print(f"Received: {line}")
            if '(Yocto Project Reference Distro) 5.2.1 agilex7_dk_si_agf014ea' in line:
                time.sleep(3)
                ser.write(b'root\n')
                break
    #For this part to work you can't be in serial because multiple people can write, but only 1 person can read at a time!
    
    print("Lemonade")

    #ser.write(b'root\n')

    time.sleep(3)

    ser.write(b'cd /usr/bin/tsi/v0.1.1.tsv32_06_20_2025/bin\n')

    ser.close()

    print("Finished Everything Hooray")

    #EVERYTHING WORKS NOW!! EVERYTHING WORKS NOW!!

@app.route('/health-check', methods=['GET'])
def health_check_serial_command():
    command = f"free -h"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/test', methods=['GET'])
def test_serial_command():
    command = f"test"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/system-info', methods=['GET'])
def system_info_serial_command():

    command = f"{exe_path}../install/tsi-version;lscpu"

    try:
        result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
        return result.stdout, 200
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {e.stderr}", 500

@app.route('/submit', methods=['POST'])
def submit():
    global job_status

    if job_status["running"]:
        return "<h2>A model is already running. Please wait or abort.</h2>"

    #./run_llama_cli.sh "my cat's name" "10" "tinyllama-vo-5m-para.gguf" "none"
    model = request.form.get('model')
    backend = request.form.get('backend')
    tokens = request.form.get('tokens')
    prompt = request.form.get('prompt')
    repeat_penalty = request.form.get('repeat-penalty', DEFAULT_REPEAT_PENALTY)
    batch_size = request.form.get('batch-size', DEFAULT_BATCH_SIZE)
    top_k = request.form.get('top-k', DEFAULT_TOP_K)
    top_p = request.form.get('top-p', DEFAULT_TOP_P)
    last_n = request.form.get('last-n', DEFAULT_LAST_N)
    context_length = request.form.get('context-length', DEFAULT_CONTEXT_LENGTH)
    temp = request.form.get('temp', DEFAULT_TEMP)

    # Define the model path (update with actual paths)
    model_paths = {
        "tiny-llama": "tinyllama-vo-5m-para.gguf",
        "Tiny-llama-F32": "Tiny-Llama-v0.3-FP32-1.1B-F32.gguf"
    }

    model_path = model_paths.get(model, "")
    if not model_path:
        return f"<h2>Error: Model path not found for '{model}'</h2>"

    # Build llama-cli command
    #command = [
    #    "./llama-cli",
    #    "-p", prompt,
    #    "-m", model_path,
    #    "--device", backend,
    #    "--temp", "0",
    #    "--n-predict", tokens,
    #    "--repeat-penalty", "1",
    #    "--top-k", "0",
    #    "--top-p", "1"
    #]

    script_path = "./run_llama_cli.sh"
    command = f"cd {exe_path}; {script_path} \"{prompt}\" {tokens} {model_path} {backend} {repeat_penalty} {batch_size} {top_k} {top_p} {last_n} {context_length} {temp}"


    def run_script():
        try:
            result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command], capture_output=True, text=True, check=True)
            job_status["result"] = result.stdout
        except subprocess.CalledProcessError as e:
            job_status["result"] = f"Error: {e.stderr}"
        finally:
            time.sleep(int(tokens)/5)
            job_status["running"] = False

    thread = threading.Thread(target=run_script)
    job_status = {"running": True, "result": "", "thread": thread}
    thread.start()

    return render_template("processing.html")

@app.route('/status')
def status():
    if job_status["running"]:
        return "running"
    else:
        return "done"

@app.route('/result')
def result():
    return render_template("result.html", output=job_status["result"])

@app.route('/abort')
def abort():
    #restart_txe_serial_command() #Put out here to avoid if loops and help testing!
    global job_status
    print(job_status["running"],job_status["thread"].is_alive())
    if job_status["running"] and job_status["thread"].is_alive():
        # Use subprocess.Popen + pid handling instead for real process termination
        job_status["running"] = False
        job_status["result"] = "Aborted by user."
        
        '''
        #THIS AUTOMATES CONTROL C OF THE SERVER BUT DOESN'T KILL THE ./LLAMA-CLI PROCESS THATS RUNNING ON PICOCOM!
        experiment = os.getpid()
        os.kill(experiment-1, signal.SIGINT)
        '''
        ser = serial.Serial(port, baudrate)

        

        ser.write(b'\x03')    # UNCOMMENT LATER BECAUSE RIGHT NOW WE HAVE TO MANUALLY REBOOT ALL THE TIME BUT THIS WORKS!
        
        ser.close()
        
        restart_txe_serial_command()
        
        #THIS WORKS NOW COMPLETLY BUT THE ONLY PROBLEM IS THAT THE WINDOW TO ABORT IS VERY INCONSISTENT
        '''
        ##################################################### EXPERIMENTAL!!!
        script_path = "./cleanup_script.sh"
        command = f"cd {exe_path}; {script_path}"

        flag =["Boo"]
        def run_script():
            try:
                flag[0] = "We went in here!"
                result = subprocess.run(['python3', 'serial_script.py', port, baudrate, command],capture_output=True,text=True)
                flag[0] = "This worked!"
                job_status["result"] = result.stdout
            except subprocess.CalledProcessError as e:
                job_status["result"] = f"Error: {e.stderr}"
            finally:
                job_status["running"] = False

        thread = threading.Thread(target=run_script)
        job_status = {"running": True, "result": "", "thread": thread}
        thread.start()
        ##################################################### EXPERIMENTAL!!!
        '''
        return "<h2>Job aborted.</h2><a href='/'>Home</a>"
    return "<h2>No job running.</h2><a href='/'>Home</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
