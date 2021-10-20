"""Error classes which may be risen."""


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class MissingInputError(Error):
    """Exception raised for errors in the input.

    Attributes
    ----------
        function_name : str
            A string containing the name of the function.

        input_names : str
            A string or a list of string with the names of the missing
            inputs.

        message : str
            Explanation of the error
    """

    def __init__(self, function_name, input_names):
        """Save the name of the function and the missing inputs."""
        self.function_name = function_name
        self.input_names = input_names

        if isinstance(input_names, str):
            self.message = (
                'The argument ' + input_names + ' is missing at the '
                + 'function ' + function_name + '!'
            )

        else:
            n = len(input_names)
            self.message = (
                'The following arguments are missing at the function '
                + function_name + ': '
            )
            for i in range(n):
                self.message = self.message + input_names[i] + ' '

        super().__init__(self.message)


class ExcessiveInputsError(Error):
    """An error exception for excessive inputs.

    Attributes
    ----------
        function_name : str
            A string with the name of the function.

        input_names : str
            A list of strings with the input names.
    """

    def __init__(self, function_name, input_names):
        """Save the name of the function and inputs."""
        self.function_name = function_name
        self.input_names = input_names
        self.message = 'You must given only one of the following inputs: '
        self.message = self.message + self.input_names[0]
        for i in range(1, len(input_names)):
            self.message = self.message + ' or ' + self.input_names[i]
        super().__init__(self.message)


class MissingAttributesError(Error):
    """Exception raised when some attribute is missing within an object.

    Attributes
    ----------
        class_name : str
            A string with the name of the class.
        attribute_name : str
            The name of the missing attribute.
    """

    def __init__(self, class_name, attribute_name):
        """Save the class of the object and the missing attribute."""
        self.class_name = class_name
        self.attribute_name = attribute_name
        super().__init__('Attribute ' + self.attribute_name + ' of class '
                         + self.class_name + ' is missing!')


class WrongTypeInput(Error):
    """Exception raised when some argument is passed in wrong type.

    Attributes
    ----------
        function_name, input_name : str
            Names of the function/method and the name of the input.

        right_type, wrong_type : str
            Names of the expected and given type, respectively.
    """

    def __init__(self, function_name, input_name, right_type, wrong_type):
        """Save the basic information."""
        self.function_name = function_name
        self.input_name = input_name
        self.right_type = right_type
        self.wrong_type = wrong_type
        self.message = ('The argument ' + input_name + ' of function '
                        + function_name + ' is expected to be ' + right_type
                        + ', not ' + wrong_type + '!')
        super().__init__(self.message)


class WrongValueInput(Error):
    """Exception raised when some argument is given with wrong value.

    Attributes
    ----------
        function_name, input_name : str
            Names of the function and input.

        expected_values : str
            A string with the rule or options of inputs.

        given_value : str
            A string with the given value impressed.
    """

    def __init__(self, function_name, input_name, expect_values, given_value):
        """Save the basic information."""
        self.function_name = function_name
        self.input_name = input_name
        self.expect_values = expect_values
        self.given_value = given_value
        self.message = ('Wrong value given for argument ' + input_name
                        + ' of function ' + function_name + '. Instead of '
                        + given_value + ', you should give according to: '
                        + expect_values)
        super().__init__(self.message)


class EmptyAttribute(Error):
    """Exception for empty class atribute.

    Attributes
    ----------
        class_name : str
        attribute_name : str
    """

    def __init__(self, class_name, attribute_name):
        """Store exception information."""
        self.class_name = class_name
        self.attribute_name = attribute_name
        self.message = ('Empty attribute ' + attribute_name + ' of class '
                        + class_name + '!')
