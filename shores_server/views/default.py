from pyramid.view import view_config


@view_config(route_name='home', renderer='shores_server:templates/mytemplate.pt')
def my_view(request):
    return {'project': 'Shores image processing server'}
