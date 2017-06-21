# Safe Image API client
Java standalone application that serves as a client for the [safe image API](https://github.com/danielbicho/SafeImage)

## Requeriments
* JDK 1.7
* Maven 3

## Build and usage
* mvn exec:java -D exec.mainClass=com.archive.common.ClientSafeImage

## Properties
File config.properties
* linkImage: Link to input image
* host: host to safeImage API

## Input Example
* linkImage=http://preprod.arquivo.pt/noFrame/replay/20110611174858im_/http://www.jornaldenegocios.pt/images/2011_01/antoniocostanot.jpg
* host=http://p27.arquivo.pt:9080/safeimage

## Output Example
* Safe[0.963098] NotSafe[0.0369021]

