class Builder():

    # Return a built environment from a configuration string
    # config_str example "mono-rect-fabric|task:flattening,observation:RGB,action:pick-and-place,initial:crumple"
    def build(config_str):
        
        # Parse the config string and call the build function
        config = Builder.parse_config_str(config_str)
        if 'fabric' in config['domain']:
            from .fabric_domain_builder \
                import FabricDomainBuilder
            return FabricDomainBuilder.build_from_config(**config)
        elif 'towel' in config['domain']:
            from .fabric_domain_builder \
                import FabricDomainBuilder
            return FabricDomainBuilder.build_from_config(**config)
        
        else:
            print("Builder: domain <{}> does not support".format(config['domain']))
            raise NotImplementedError

    def parse_config_str(config_str):
        config = {'domain': config_str.split('|')[0]}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v


        config['num_picker'] = 2
        if '(' in config['action']:
            config['num_picker'] = int(config['action'].split('(')[1].split(')')[0])
        
        config['org_action'] = config['action']
        config['action'] = config['action'].split('(')[0]

        return config