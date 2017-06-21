package com.archive.common;


import java.io.FileInputStream;
import java.io.InputStream;
import java.net.URL;
import java.util.Properties;

import org.apache.commons.io.IOUtils;
import org.json.JSONObject;

import com.sun.jersey.api.client.Client;
import com.sun.jersey.api.client.ClientResponse;
import com.sun.jersey.api.client.WebResource;
import com.sun.org.apache.xerces.internal.impl.dv.util.Base64;


public class ClientSafeImage {
	
	public static String pathProperties = "config.properties";
	public static Properties prop;
	
	public static char loadConfigs( ) {
		InputStream input;
		try {
			input = new FileInputStream( pathProperties );
			prop = new Properties( );
			// load a properties file
			prop.load( input );
			// get the property value and print it out
			System.out.println( "image teste == " + prop.getProperty( "linkImage" ) );
			System.out.println( "host safeImage API == " + prop.getProperty( "host" ) );
			if( prop == null || prop.getProperty( "host" ) == null || prop.getProperty( "host" ).equals( "" ) || prop.getProperty( "linkImage" ) == null || prop.getProperty( "linkImage" ).equals( "" ) )
				return 46;
		} catch( Exception e ) {
			System.out.println( e );
			return 45;
		}
		return 46;
	}
	
	public static void main( String[ ] args ) {

		try {
			if( loadConfigs(  ) == 45 ) 
				return;
			int valor = 2097152;
			System.out.println( "valor => " + valor );
			Client client = Client.create( );
			WebResource webResource = client
					.resource( prop.getProperty( "host" ) );
			byte[ ] bytesImgOriginal = IOUtils.toByteArray( new URL( prop.getProperty( "linkImage" ) ).openConnection( ).getInputStream( ) );
			
			String base64String = Base64.encode( bytesImgOriginal );
			JSONObject input = new JSONObject( );
			input.put("image", base64String );
			System.out.println( "input => " + input.toString( ) );
			ClientResponse response = webResource.type( "application/json" )
					.post( ClientResponse.class, input.toString( ) );

			if ( response.getStatus( ) != 200 ) {
				throw new RuntimeException( "Failed : HTTP error code : "
						+ response.getStatus( ) );
			}

			System.out.println( "Output from Server ...." );
			String outputJS = response.getEntity( String.class );
			System.out.println( "outputJS => " + outputJS );
			//JSONObject output = new JSONObject(  outputJS.substring( 1, outputJS.length( ) - 2 ).replace( "\\" , "" ) ); 
			JSONObject output = new JSONObject( outputJS );
			String NSFW = (String) output.get( "NSFW" );
			System.out.println( "NSFW[" + NSFW + "]\n\n" );

		} catch ( Exception e ) {
			e.printStackTrace();
		}

	}

}
