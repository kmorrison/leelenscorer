SEP = b'\n\n\n\n'


def write_payload(writer, payload):
    for line in payload:
        writer.write(line)
        writer.write(SEP)


def remove_sep(bytestring):
    if not bytestring:
        return bytestring
    if bytestring[-1] == SEP[-1]:
        return bytestring[:len(bytestring) - len(SEP)]
    else:
        return bytestring
