"""
Execution utilities.
"""
from subprocess import Popen, PIPE, call
import logging
from math import floor
import os
import psutil
import getpass
import shlex
from warnings import warn


logger = logging.getLogger(__name__)


def log_mem():
    """Print memory status to debug logger."""
    mem = psutil.virtual_memory()
    logger.debug('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
                 '({3:.3f} GB free) ({4:.3f} GB available).'
                 ''.format(mem.used / 1e9,
                           mem.total / 1e9,
                           100 * mem.used / mem.total,
                           mem.free / 1e9,
                           mem.available / 1e9))


class SubprocessManager:
    """Base class to handle subprocess execution."""

    # get username as class attribute.
    USER = getpass.getuser()

    @staticmethod
    def make_path(d):
        """Make a directory tree if it doesn't exist.

        Parameters
        ----------
        d : str
            Directory tree to check and potentially create.
        """
        if not os.path.exists(d):
            os.makedirs(d)

    @staticmethod
    def make_sh(fname, script):
        """Make a shell script (.sh file) to execute a subprocess.

        Parameters
        ----------
        fname : str
            Name of the .sh file to create.
        script : str
            Contents to be written into the .sh file.
        """
        logger.debug('The shell script "{n}" contains the following:\n'
                     '~~~~~~~~~~ {n} ~~~~~~~~~~\n'
                     '{s}\n'
                     '~~~~~~~~~~ {n} ~~~~~~~~~~'
                     .format(n=fname, s=script))
        with open(fname, 'w+') as f:
            f.write(script)

    @staticmethod
    def rm(fname):
        """Remove a file.

        Parameters
        ----------
        fname : str
            Filename (with path) to remove.
        """
        os.remove(fname)

    @staticmethod
    def submit(cmd):
        """Open a subprocess and submit a command.

        Parameters
        ----------
        cmd : str
            Command to be submitted using python subprocess.

        Returns
        -------
        stdout : str
            Subprocess standard output. This is decoded from the subprocess
            stdout with rstrip.
        stderr : str
            Subprocess standard error. This is decoded from the subprocess
            stderr with rstrip. After decoding/rstrip, this will be empty if
            the subprocess doesn't return an error.
        """

        cmd = shlex.split(cmd)

        # use subprocess to submit command and get piped o/e
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stderr = stderr.decode('ascii').rstrip()
        stdout = stdout.decode('ascii').rstrip()

        if stderr:
            raise Exception('Error occurred submitting job:\n{}'
                            .format(stderr))

        return stdout, stderr

    @staticmethod
    def s(s):
        """Format input as str w/ appropriate quote types for python cli entry.

        Examples
        --------
            list, tuple -> "['one', 'two']"
            dict -> "{'key': 'val'}"
            int, float, None -> '0'
            str, other -> 'string'
        """

        if isinstance(s, (list, tuple, dict)):
            return '"{}"'.format(s)
        elif not isinstance(s, (int, float, type(None))):
            return "'{}'".format(s)
        else:
            return '{}'.format(s)

    @staticmethod
    def walltime(hours):
        """Get the SLURM walltime string in format "HH:MM:SS"

        Parameters
        ----------
        hours : float | int
            Requested number of job hours.

        Returns
        -------
        walltime : str
            SLURM walltime request in format "HH:MM:SS"
        """

        m_str = '{0:02d}'.format(round(60 * (hours % 1)))
        h_str = '{0:02d}'.format(floor(hours))
        return '{}:{}:00'.format(h_str, m_str)


class PBS(SubprocessManager):
    """Subclass for PBS subprocess jobs."""

    def __init__(self, cmd, alloc, queue, name='nsrdb',
                 feature=None, stdout_path='./stdout'):
        """Initialize and submit a PBS job.

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m nsrdb.clu'
        alloc : str
            HPC allocation account. Example: 'prsc'.
        queue : str
            HPC queue to submit job to. Example: 'short', 'batch-h', etc...
        name : str
            PBS job name.
        feature : str | None
            PBS feature request (-l {feature}).
            Example: 'feature=24core', 'qos=high', etc...
        stdout_path : str
            Path to print .stdout and .stderr files.
        """

        self.make_path(stdout_path)
        self.id, self.err = self.qsub(cmd,
                                      alloc=alloc,
                                      queue=queue,
                                      name=name,
                                      feature=feature,
                                      stdout_path=stdout_path)

    def check_status(self, job, var='id'):
        """Check the status of this PBS job using qstat.

        Parameters
        ----------
        job : str
            Job name or ID number.
        var : str
            Identity/type of job identification input arg ('id' or 'name').

        Returns
        -------
        out : str or NoneType
            Qstat job status character or None if not found.
            Common status codes: Q, R, C (queued, running, complete).
        """

        # column location of various job identifiers
        col_loc = {'id': 0, 'name': 3}
        qstat_rows = self.qstat()
        if qstat_rows is None:
            return None
        else:
            # reverse the list so most recent jobs are first
            qstat_rows = reversed(qstat_rows)

        # update job status from qstat list
        for row in qstat_rows:
            row = row.split()
            # make sure the row is long enough to be a job status listing
            if len(row) > 10:
                if row[col_loc[var]].strip() == job.strip():
                    # Job status is located at the -2 index
                    status = row[-2]
                    logger.debug('Job with {} "{}" has status: "{}"'
                                 .format(var, job, status))
                    return status
        return None

    def qstat(self):
        """Run the PBS qstat command and return the stdout split to rows.

        Returns
        -------
        qstat_rows : list | None
            List of strings where each string is a row in the qstat printout.
            Returns None if qstat is empty.
        """

        cmd = 'qstat -u {user}'.format(user=self.USER)
        stdout, _ = self.submit(cmd)
        if not stdout:
            # No jobs are currently running.
            return None
        else:
            qstat_rows = stdout.split('\n')
            return qstat_rows

    def qsub(self, cmd, alloc, queue, name='nsrdb', feature=None,
             stdout_path='./stdout', keep_sh=False):
        """Submit a PBS job via qsub command and PBS shell script

        Parameters
        ----------
        cmd : str
            Command to be submitted in PBS shell script. Example:
                'python -m nsrdb.cli'
        alloc : str
            HPC allocation account. Example: 'nsrdb'.
        queue : str
            HPC queue to submit job to. Example: 'short', 'batch-h', etc...
        name : str
            PBS job name.
        feature : str | None
            PBS feature request (-l {feature}).
            Example: 'feature=24core', 'qos=high', etc...
        stdout_path : str
            Path to print .stdout and .stderr files.
        keep_sh : bool
            Boolean to keep the .sh files. Default is to remove these files
            after job submission.

        Returns
        -------
        out : str
            qsub standard output, this is typically the PBS job ID.
        err : str
            qsub standard error, this is typically an empty string if the job
            was submitted successfully.
        """

        status = self.check_status(name, var='name')

        if status == 'Q' or status == 'R':
            warn('Not submitting job "{}" because it is already in '
                 'qstat with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'
        else:
            feature_str = '#PBS -l {}\n'.format(feature)
            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#PBS -N {n} # job name\n'
                      '#PBS -A {a} # allocation account\n'
                      '#PBS -q {q} # queue (debug, short, batch, or long)\n'
                      '#PBS -o {p}/{n}_$PBS_JOBID.o\n'
                      '#PBS -e {p}/{n}_$PBS_JOBID.e\n'
                      '{L}'
                      'echo Running on: $HOSTNAME, Machine Type: $MACHTYPE\n'
                      '{cmd}'
                      .format(n=name, a=alloc, q=queue, p=stdout_path,
                              L=feature_str if feature else '',
                              cmd=cmd))

            # write the shell script file and submit as qsub job
            self.make_sh(fname, script)
            out, err = self.submit('qsub {script}'.format(script=fname))

            if not err:
                logger.debug('PBS job "{}" with id #{} submitted successfully'
                             .format(name, out))
                if not keep_sh:
                    self.rm(fname)

        return out, err


class SLURM(SubprocessManager):
    """Subclass for SLURM subprocess jobs."""

    def __init__(self, cmd, alloc, memory, walltime, feature='--qos=normal',
                 name='nsrdb', stdout_path='./stdout'):
        """Initialize and submit a PBS job.

        Parameters
        ----------
        cmd : str
            Command to be submitted in SLURM shell script.
        alloc : str
            HPC project (allocation) handle. Example: 'pxs'.
        memory : int
            Node memory request in GB.
        walltime : float
            Node walltime request in hours.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]". Default is None.
        name : str
            SLURM job name.
        stdout_path : str
            Path to print .stdout and .stderr files.
        """

        self.make_path(stdout_path)
        self.out, self.err = self.sbatch(cmd,
                                         alloc=alloc,
                                         memory=memory,
                                         walltime=walltime,
                                         feature=feature,
                                         name=name,
                                         stdout_path=stdout_path)
        if self.out:
            self.id = self.out.split(' ')[-1]
        else:
            self.id = None

    @staticmethod
    def check_status(job, var='id'):
        """Check the status of this PBS job using qstat.

        Parameters
        ----------
        job : str
            Job name or ID number.
        var : str
            Identity/type of job identification input arg ('id' or 'name').

        Returns
        -------
        out : str | NoneType
            squeue job status str or None if not found.
            Common status codes: PD, R, CG (pending, running, complete).
        """

        # column location of various job identifiers
        col_loc = {'id': 0, 'name': 2}

        if var == 'name':
            # check for specific name
            squeue_rows = SLURM.squeue(name=job)
        else:
            squeue_rows = SLURM.squeue()

        if squeue_rows is None:
            return None
        else:
            # reverse the list so most recent jobs are first
            squeue_rows = reversed(squeue_rows)

        # update job status from qstat list
        for row in squeue_rows:
            row = row.split()
            # make sure the row is long enough to be a job status listing
            if len(row) > 7:
                if row[col_loc[var]].strip() in job.strip():
                    # Job status is located at the 4 index
                    status = row[4]
                    logger.debug('Job with {} "{}" has status: "{}"'
                                 .format(var, job, status))
                    return row[4]
        return None

    @staticmethod
    def squeue(name=None):
        """Run the SLURM squeue command and return the stdout split to rows.

        Parameters
        ----------
        name : str | None
            Optional to check the squeue for a specific job name (not limited
            to the 8 shown characters) or show users whole squeue.

        Returns
        -------
        squeue_rows : list | None
            List of strings where each string is a row in the squeue printout.
            Returns None if squeue is empty.
        """

        cmd = ('squeue -u {user}{job_name}'
               .format(user=SLURM.USER,
                       job_name=' -n {}'.format(name) if name else ''))
        stdout, _ = SLURM.submit(cmd)
        if not stdout:
            # No jobs are currently running.
            return None
        else:
            squeue_rows = stdout.split('\n')
            return squeue_rows

    @staticmethod
    def scancel(job_id):
        """Cancel a slurm job.

        Parameters
        ----------
        job_id : int
            SLURM job id to cancel
        """

        cmd = ('scancel {job_id}'.format(job_id=job_id))
        cmd = shlex.split(cmd)
        call(cmd)

    @staticmethod
    def _scancel_all():
        """Cancel all user jobs.

        Parameters
        ----------
        job_id : int
            SLURM job id to cancel
        """
        squeue_rows = SLURM.squeue()
        for row in squeue_rows[1:]:
            cmd = ('scancel {job_id}'.format(job_id=row.strip().split(' ')[0]))
            cmd = shlex.split(cmd)
            call(cmd)

    def sbatch(self, cmd, alloc, walltime, memory=None, feature='--qos=normal',
               name='nsrdb', stdout_path='./stdout', keep_sh=False):
        """Submit a SLURM job via sbatch command and SLURM shell script

        Parameters
        ----------
        cmd : str
            Command to be submitted in SLURM shell script.
        alloc : str
            HPC project (allocation) handle. Example: 'pxs'.
        walltime : float
            Node walltime request in hours.
        memory : int
            Node memory request in GB.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]". Default is None.
        name : str
            SLURM job name.
        stdout_path : str
            Path to print .stdout and .stderr files.
        keep_sh : bool
            Boolean to keep the .sh files. Default is to remove these files
            after job submission.

        Returns
        -------
        out : str
            sbatch standard output, this is typically the SLURM job ID.
        err : str
            sbatch standard error, this is typically an empty string if the job
            was submitted successfully.
        """

        status = self.check_status(name, var='name')

        if status in ('PD', 'R'):
            warn('Not submitting job "{}" because it is already in '
                 'squeue with status: "{}"'.format(name, status))
            out = None
            err = 'already_running'

        else:

            feature_str = ''
            if feature is not None:
                feature_str = '#SBATCH {}  # extra feature\n'.format(feature)

            mem_str = ''
            if memory is not None:
                mem_str = ('#SBATCH --mem={}  # node RAM in MB\n'
                           .format(int(memory * 1000)))

            fname = '{}.sh'.format(name)
            script = ('#!/bin/bash\n'
                      '#SBATCH --account={a}  # allocation account\n'
                      '#SBATCH --time={t}  # walltime\n'
                      '#SBATCH --job-name={n}  # job name\n'
                      '#SBATCH --nodes=1  # number of nodes\n'
                      '#SBATCH --output={p}/{n}_%j.o\n'
                      '#SBATCH --error={p}/{n}_%j.e\n{m}{f}'
                      'echo Running on: $HOSTNAME, Machine Type: $MACHTYPE\n'
                      '{cmd}'
                      .format(a=alloc, t=self.walltime(walltime), n=name,
                              p=stdout_path, m=mem_str, f=feature_str,
                              cmd=cmd))

            # write the shell script file and submit as qsub job
            self.make_sh(fname, script)
            out, err = self.submit('sbatch {script}'.format(script=fname))

            if not err:
                logger.debug('SLURM job "{}" with id #{} submitted '
                             'successfully'.format(name, out))
                if not keep_sh:
                    self.rm(fname)

        return out, err
