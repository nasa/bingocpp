#include <string>
#include <vector>
#include <unordered_map>

namespace bingo {

typedef std::unordered_map<int, std::string> PrintMap;
typedef std::vector<std::vector<std::string>> PrintVector;

const PrintMap kStackPrintMap {
  {2, "({}) + ({})"},
  {3, "({}) - ({})"},
  {4, "({}) * ({})"},
  {5, "({}) / ({}) "},
  {6, "sin ({})"},
  {7, "cos ({})"},
  {8, "exp ({})"},
  {9, "log ({})"},
  {10, "({}) ^ ({})"},
  {11, "abs ({})"},
  {12, "sqrt ({})"},
};

const PrintMap kLatexPrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "\\frac{ {} }{ {} }"},
  {6, "sin{ {} }"},
  {7, "cos{ {} }"},
  {8, "exp{ {} }"},
  {9, "log{ {} }"},
  {10, "({})^{ ({}) }"},
  {11, "|{}|"},
  {12, "\\sqrt{ {} }"},
};

const PrintMap kConsolePrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "({})/({})"},
  {6, "sin({})"},
  {7, "cos({})"},
  {8, "exp({})"},
  {9, "log({})"},
  {10, "({})^({})"},
  {11, "|{}|"},
  {12, "sqrt({})"},
};

const bool kIsArity2Map[13] = {
  false,
  false,
  true,
  true,
  true,
  true,
  false,
  false,
  false,
  false,
  true,
  false,
  false
};

const bool kIsTerminalMap[13] = {
  true,
  true,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false
};

const PrintVector kOperatorNames {
  std::vector<std::string> {"load", "x"},
  std::vector<std::string> {"constant", "c"},
  std::vector<std::string> {"add", "addition", "+"},
  std::vector<std::string> {"subtract", "subtraction", "-"},
  std::vector<std::string> {"multiply", "multiplication", "*"},
  std::vector<std::string> {"divide", "division", "/"},
  std::vector<std::string> {"sine", "sin"},
  std::vector<std::string> {"cosine", "cos"},
  std::vector<std::string> {"exponential", "exp", "e"},
  std::vector<std::string> {"logarithm", "log"},
  std::vector<std::string> {"power", "pow", "^"},
  std::vector<std::string> {"absolute value", "||", "|"},
  std::vector<std::string> {"square root", "sqrt"}
};
} // namespace bingo
