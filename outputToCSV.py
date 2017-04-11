def flatten_results(results):
    flat_results = []
    for result in results:
        flat_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat_result[str(key) + "_" + str(k)] = v
            else:
                flat_result[key] = value
        flat_results.append(flat_result)
    return flat_results

def out_to_csv(col_names, flat_results, filename):
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=col_names, dialect='excel')
        writer.writeheader()
        for result in flat_results:
            writer.writerow(result)
            
def in_from_csv(filename):
    import csv
    flat_results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, dialect='excel')
        flat_results = list(reader)
    return (flat_results)

def unflatten_results(flat_results):
    results = []
    for flat_result in flat_results:
        result = {'chunky_result':{},'local_result':{},'dw_result':{}}
        for key, value in flat_result.items():
            if key[0:6] == "chunky":
                result['chunky_result'][key[14:]] = value
            elif key[0:5] == "local":
                result['local_result'][key[13:]] = value
            elif key[0:2] == "dw":
                result['dw_result'][key[10:]] = value
            else:
                result[key] = value
        results.append(result)
    return results
                
        