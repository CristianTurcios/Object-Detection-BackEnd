## Development server

 Este proyecto utiliza flask para servir una api y que el FE lo pueda consultar.

 Para correr este proyecto es necesario ejecutar el siguiente comando: python `app.py`. Esto iniciara un servidor de flask en el puerto `1005` el cual tiene un endpoint que recibe una imagen, la procesa con un modelo entrenado y devuelve las predicciones

El endpoint que se puede llamar es:

`POST`: `/detection`

## Tecnologias

  Las tecnologias que utiliza este proyecto son

- Python
- Flask