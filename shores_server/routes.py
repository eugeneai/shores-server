from .views.rest import post_sparql

def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('sparql', '/sa-1.0/sparql/{img_uuid}/query/')
    config.add_view(post_sparql, route_name='sparql', request_method="POST")
   # config.scan('.views.rest')
