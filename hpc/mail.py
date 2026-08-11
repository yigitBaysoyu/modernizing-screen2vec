import os
import sys
import smtplib
import shutil
import zipfile
import glob
import subprocess
from email.message import EmailMessage


def get_txt_files(directory):
    """Retrieve all .txt files in the given directory."""
    return [file for file in os.listdir(directory) if file.endswith('.txt') and ('stdout' in file or 'stderr' in file or 'output' in file)]

def zip_folders_with_prefix(directory, prefix):
    """Zip all folders in the given directory that start with the specified prefix."""
    zip_files = []
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path) and folder.startswith(prefix):
            zip_filename = f"{folder}.zip"
            zip_filepath = os.path.join(directory, zip_filename)
            # Note: Removed compresslevel due to compatibility issues
            shutil.make_archive(base_name=zip_filepath.replace('.zip', ''), format='zip', root_dir=folder_path, base_dir='.')
            zip_files.append(zip_filepath)
    return zip_files

def find_adapter_output_dir():
    """
    Locate the checkpoint directory of an adapter training job, or None if this
    job is not one.

    The directory name is not fixed any more: every sweep point of
    2_train_adapter.py writes its own (adapter_model_out_embeddings_4096_last_full,
    ...), so the submitting start script exports $ADAPTER_OUTPUT_DIR and we read
    it here.  "adapter_model_out" stays as a fallback for jobs submitted with an
    older start.sh.

    Deliberately no glob over adapter_model_out*: the task folder is a copy of
    the whole project and usually still carries the checkpoint directories of
    *previous* runs, so a glob would make every reconstruction/evaluation job
    mail stale checkpoints back.
    """
    explicit = os.environ.get("ADAPTER_OUTPUT_DIR", "").strip()
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        print(
            f"ADAPTER_OUTPUT_DIR is set to '{explicit}', but no such directory "
            f"exists in {os.getcwd()}. No checkpoints will be sent.",
            file=sys.stderr,
        )
        return None

    return "adapter_model_out" if os.path.isdir("adapter_model_out") else None


def send_email_with_attachments(subject, body, attachments):
    """Send an email with attachments."""
    # Create the email message
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "jonathan.liebers+amplitude@uni-duisburg-essen.de"
    msg['To'] = "jonathan.liebers@uni-due.de"
    msg['Cc'] = "amplitude@jo2k.de"
    msg.set_content(body)

    # Add attachments
    for filepath in attachments:
        with open(filepath, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(filepath)
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    # Send the email
    with smtplib.SMTP('mailout.uni-due.de', 587, timeout=120) as server:
        server.starttls()
        server.send_message(msg)

def main():
    subject = f'amplitUDE output for job {sys.argv[1:]}'

    # Get list of all .txt files in the current directory
    current_directory = os.getcwd()
    txt_files = get_txt_files(current_directory)

    # Delete all pickle files that should not go into the mail
    os.system("find -name '*pickle' -delete")
    os.system("find -name '*pkl' -delete")
    os.system("find -name '*hdf5' -delete")
    os.system("find -name 'W*txt' -delete")

    # Load output (merged or stdout/stderr) for email body
    try:
        job_id = sys.argv[1] if len(sys.argv) > 1 else ""
        if os.path.exists(f"hpc/output_{job_id}.txt"):
            with open(f"hpc/output_{job_id}.txt") as f:
                body = f.read()
        elif os.path.exists(f"output_{job_id}.txt"):
            with open(f"output_{job_id}.txt") as f:
                body = f.read()
        else:
            with open(f"hpc/stdout_{job_id}.txt") as stdout_file, open(f"hpc/stderr_{job_id}.txt") as stderr_file:
                body = stdout_file.read() + "\n=============================== \n\n" + stderr_file.read()
    except Exception as e:
        print(f"Could not read output logs: {e}", file=sys.stderr)
        body = "Log files could not be read."

    # Check if this is an adapter training job (meaning a checkpoint dir exists)
    adapter_dir = find_adapter_output_dir()
    if adapter_dir:
        folder_name = os.path.basename(current_directory)
        print(f"Detected adapter output directory '{adapter_dir}' in {folder_name}. Preparing split bzip2 archive...")
        archive_name = "../model.zip"
        
        # Clean up any existing archives first
        for f in glob.glob("model.z*") + ["model.zip"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        try:
            # Create split zip archive inside the checkpoint dir with highest bzip2 compression, saving to parent directory
            cmd = ["zip", "-s", "25m", "-r", "-9", "-Z", "bzip2", archive_name, "."]
            print(f"Running command: {' '.join(cmd)} (in {adapter_dir})")
            subprocess.run(cmd, cwd=adapter_dir, check=True)
        except Exception as e:
            print(f"Error creating split bzip2 archive: {e}", file=sys.stderr)

        # Find all generated chunks in current directory
        splits = glob.glob("model.z[0-9]*")
        splits.sort()
        all_chunks = splits
        if os.path.exists("model.zip"):
            all_chunks.append("model.zip")

        if not all_chunks:
            print("No split archive chunks found to send.", file=sys.stderr)
            return

        total_chunks = len(all_chunks)
        print(f"Sending {total_chunks} chunks as individual emails...")
        for idx, chunk in enumerate(all_chunks):
            part_num = idx + 1
            subject_part = f"{subject} ({folder_name}/{adapter_dir}) - Part {part_num}/{total_chunks}"
            
            # Attach txt_files (logs) only to the first chunk email to avoid duplicating large logs in every email
            if part_num == 1:
                attachments = [chunk] + txt_files
            else:
                attachments = [chunk]
                
            print(f"Sending email for chunk {part_num}/{total_chunks}: {chunk}")
            send_email_with_attachments(subject_part, body, attachments)

        # Clean up the archive files
        for chunk in all_chunks:
            try:
                os.remove(chunk)
            except Exception as e:
                print(f"Error removing archive chunk {chunk}: {e}", file=sys.stderr)
                
        print("Finished sending all adapter archive emails.")

    else:
        # Standard useckit / csv flow
        # Zip all folders that start with "_useckit"
        zip_files = zip_folders_with_prefix(current_directory, "_useckit")

        # Zip all .csv files found in the current directory into one archive
        csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]
        if csv_files:
            csv_zip_filename = "csv_results.zip"
            with zipfile.ZipFile(csv_zip_filename, 'w', zipfile.ZIP_LZMA) as zipf:
                for csv_file in csv_files:
                    zipf.write(csv_file)
            zip_files.append(csv_zip_filename)

        # Combine txt files and zipped folders for email attachments
        attachments = txt_files + zip_files

        # Send email
        if attachments:
            send_email_with_attachments(subject, body, attachments)
            print("Email sent successfully with attachments:", attachments)
            
            # Remove the files after sending the email
            for file in attachments:
                try:
                    # Do not delete log files
                    if 'stdout' in file or 'stderr' in file or 'output' in file:
                        continue
                    os.remove(file)
                except FileNotFoundError as f:
                    print(f, "CWD is", os.getcwd(), file=sys.stderr)
        else:
            print("No attachments found in the current directory.")

if __name__ == "__main__":
    os.chdir("..")
    main()
    print("Finished.")