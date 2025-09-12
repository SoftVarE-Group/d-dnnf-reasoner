# Python bindings for `ddnnife`

## Usage

d-DNNFs are loaded via the `Ddnnf` class and its `from_file` method.
Some methods are available directly on a `Ddnnf`, while others require a mutable version.
A `Ddnnf` can be made mutable with `as_mut()`.

## Example

```python
from ddnnife import Ddnnf

ddnnf = Ddnnf.from_file("path/to/ddnnf", None)
count = ddnnf.as_mut().count([1, 2, 3])
print(count)
```
