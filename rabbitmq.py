import pika
import sys


def send_message(json):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='data_queue', durable=False)

    #message = ' '.join(sys.argv[1:]) or "Hello World!"
    channel.basic_publish(
        exchange='',
        routing_key='data_queue',
        body=json,
        properties=pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
        ))
    print(" [x] Sent %r" % json)
    connection.close()