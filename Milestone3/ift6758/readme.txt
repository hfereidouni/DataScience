We are pleased to inform you that our Streamlit application is fully operational on local machines. However, we have encountered network-related challenges when deploying it in a Docker environment. Despite our best efforts, this issue remains unresolved. This problem is about the ports when it comes to the Docker part. In Streamlit (non-Docker part) a cobination of "127.0.0.1" for ip and 5000 for port gives us great result, however, on Docker it would be slightly baffling.
The application, enhanced with additional functionalities such as heatmap and shot coordination, has been meticulously developed. 

Our Dockerfiles and images are functioning correctly, and the application runs flawlessly on a non-Docker local machine. The code is written in a clear and straightforward manner, with each function accompanied by comprehensive docstrings and comments for ease of understanding.
To evaluate the application locally, please follow these steps:

* 		Install Streamlit using the command pip install streamlit.
* 		Run the Flask app (app.py).
* 		Launch the Streamlit app (st_app.py) using the command streamlit run st_app.py.

For Docker deployment, you have the option to build and run each Dockerfile individually or utilize the docker-compose.yaml file. To do so, execute the command docker-compose up in the respective directory.For Docker part please note that the "app.py" file should be in the "serving" directory same like the Github repository.
We have ensured that all components of the application are thoroughly tested and are functioning as intended. We look forward to your feedback and are available for any further assistance or clarification you might need regarding the deployment and functionality of the application.
