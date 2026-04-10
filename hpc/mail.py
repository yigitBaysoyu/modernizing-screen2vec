import os
import sys
import smtplib
import shutil
from email.message import EmailMessage


def get_txt_files(directory):
    """Retrieve all .txt files in the given directory."""
    return [file for file in os.listdir(directory) if file.endswith('.txt') and ('stdout' in file or 'stderr' in file)]

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

    # Zip all folders that start with "_useckit"
    zip_files = zip_folders_with_prefix(current_directory, "_useckit")

    # set body to stdout
    with open(f"hpc/stderr_{sys.argv[1]}.txt") as stderr_file, open(f"hpc/stdout_{sys.argv[1]}.txt") as stdout_file:
        body = stderr_file.read() + "\n=============================== \n\n" + stdout_file.read()

    # Combine txt files and zipped folders for email attachments
    attachments = txt_files + zip_files

    # Send email
    if attachments:
        send_email_with_attachments(subject, body, attachments)
        print("Email sent successfully with attachments:", attachments)
        
        # Remove the files after sending the email
        for file in attachments:
            try:
                os.remove(file)
                os.remove(f"hpc/stderr_{sys.argv[1]}.txt")
                os.remove(f"hpc/stdout_{sys.argv[1]}.txt")
            except FileNotFoundError as f:
                print(f, "CWD is", os.getcwd(), file=sys.stderr)
    else:
        print("No attachments found in the current directory.")

if __name__ == "__main__":
    os.chdir("..")
    main()
    print("Finished.")
