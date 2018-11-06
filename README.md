# amine-team || BMIAI Hackathon Code

## Development setup

SSH into the ECE remote:
```
ssh root@kaggle3.ece.ubc.ca
```

To run the Jupyter notebook on the ECE remote:
```
jupyter notebook --no-browser --port=1234
```

The output of this command will show you a url that has a token parameter
somewhere:
```
[I 21:35:19.309 NotebookApp] The Jupyter Notebook is running at:
[I 21:35:19.309 NotebookApp] http://(kaggle3.ece.ubc.ca or 127.0.0.1):1234/?token=<YOUR TOKEN IS HERE>
[I 21:35:19.309 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 21:35:19.309 NotebookApp] 

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://(kaggle3.ece.ubc.ca or 127.0.0.1):1234/?token=<YOUR TOKEN IS HERE>
```

Then create an SSH tunnel between your local machine and the Jupyter server by
running the following command on your local machine:
```
ssh -NfL localhost:5555:localhost:1234 root@kaggle3.ece.ubc.ca
```

After running that command, you should be able to open up the Jupyter notebook
at `localhost:5555` on your local machine. On the first run, Jupyter will ask
you for a token. Copy paste the token from a previous step into the input field,
and you should be good to go.
